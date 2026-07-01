"""CompanyClassifierService — classify a profile's style + model stack, publish."""

from datetime import UTC, datetime

import structlog
from trading_common.events import CompanyClassifiedEvent
from trading_common.schemas import CompanyProfile

from src.core.classifier import ClassificationResult, ValuationMetrics, classify
from src.events.publisher import Publisher

logger = structlog.get_logger()


class CompanyClassifierService:
    def __init__(self, publisher: Publisher) -> None:
        self._publisher = publisher
        # latest enriched profile + its classification, per symbol
        self._latest: dict[str, tuple[CompanyProfile, ClassificationResult]] = {}

    def get(self, symbol: str) -> tuple[CompanyProfile, ClassificationResult] | None:
        return self._latest.get(symbol.upper())

    def symbols(self) -> list[str]:
        return sorted(self._latest)

    async def classify(
        self, profile: CompanyProfile, metrics: ValuationMetrics | None = None
    ) -> tuple[CompanyProfile, ClassificationResult]:
        """Assign style + model stack to a company profile and publish the result."""
        result = classify(profile.sector, profile.market_cap, metrics)
        enriched = profile.model_copy(
            update={
                "style": result.style,
                "model_stack": result.model_stack,
                "as_of": datetime.now(UTC),
            }
        )
        self._latest[enriched.symbol.upper()] = (enriched, result)

        await self._publisher.publish(
            CompanyClassifiedEvent(
                symbol=enriched.symbol,
                style=result.style,
                model_stack=result.model_stack,
            )
        )
        logger.info(
            "Company classified",
            symbol=enriched.symbol,
            style=result.style,
            model_stack=result.model_stack,
            cap_tier=result.cap_tier,
            basis=result.basis,
        )
        return enriched, result
