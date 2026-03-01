{{/*
Nazwa obrazu serwisu.
Użycie: {{ include "trading.image" (dict "values" .Values.marketData "global" .Values.global) }}
*/}}
{{- define "trading.image" -}}
{{ .global.imageRegistry }}/{{ .values.image.name }}:{{ .values.image.tag | default "latest" }}
{{- end }}

{{/*
Standardowe labels dla wszystkich zasobów.
*/}}
{{- define "trading.labels" -}}
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
environment: {{ .Values.global.env }}
{{- end }}

{{/*
Prometheus scrape annotations.
*/}}
{{- define "trading.prometheusAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: "8000"
prometheus.io/path: "/metrics"
{{- end }}
