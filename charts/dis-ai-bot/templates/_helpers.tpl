{{/*
Expand the name of the chart.
*/}}
{{- define "helm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "helm.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "helm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "helm.labels" -}}
helm.sh/chart: {{ include "helm.chart" . }}
{{ include "helm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "helm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "helm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "helm.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "helm.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
The local LLM server this release runs (llmEngine): llamacpp or strata.
Anything else fails the render rather than quietly running neither.
*/}}
{{- define "dcb.llmEngine" -}}
{{- $engine := .Values.llmEngine | default "llamacpp" }}
{{- if not (has $engine (list "llamacpp" "strata")) }}
{{- fail (printf "llmEngine must be llamacpp or strata, not %q" $engine) }}
{{- end }}
{{- $engine }}
{{- end }}

{{/*
core's LLM_HOST: llm_host when set, else the in-cluster service of the
selected engine. An explicit llm_host is kept as-is, so it must be changed
too when switching engines.
*/}}
{{- define "dcb.llmHost" -}}
{{- if .Values.llm_host }}
{{- .Values.llm_host }}
{{- else if eq (include "dcb.llmEngine" .) "strata" }}
{{- printf "http://%s-strata-service.%s.svc.cluster.local:%v" .Release.Name .Release.Namespace .Values.strata.port }}
{{- else }}
{{- printf "http://%s-llamacpp-service.%s.svc.cluster.local:%v" .Release.Name .Release.Namespace .Values.llamacpp.port }}
{{- end }}
{{- end }}

{{/*
The Secret holding Strata's API key (key STRATA_API_KEY): read by the Strata
pod (API_KEY) and by core (LLM_PASS), so the two can never disagree. Strata
always runs with a key; without one of ours it would make its own, which
core would never see, so the render fails instead.
*/}}
{{- define "dcb.strataSecretName" -}}
{{- if .Values.strata.existingSecret }}
{{- .Values.strata.existingSecret }}
{{- else if .Values.strata.apiKey }}
{{- printf "%s-strata-secret" .Release.Name }}
{{- else }}
{{- fail "llmEngine=strata needs strata.apiKey or strata.existingSecret (key STRATA_API_KEY): Strata will not serve without an API key" }}
{{- end }}
{{- end }}

{{/*
LLM_CONTEXT_LENGTH: the server's TOTAL context. llama.cpp's
llamacppContextLength is already that (it splits it across slots); Strata's
strata.context is per slot (32k = 32768 tokens), so it is multiplied by
strata.parallel (empty, i.e. Strata's choice, counts as 1).
*/}}
{{- define "dcb.llmContextLength" -}}
{{- if eq (include "dcb.llmEngine" .) "strata" }}
{{- $context := .Values.strata.context | toString | lower | trim }}
{{- $perSlot := 0 }}
{{- if hasSuffix "k" $context }}
{{- $perSlot = mul (trimSuffix "k" $context | atoi) 1024 }}
{{- else }}
{{- $perSlot = atoi $context }}
{{- end }}
{{- if le $perSlot 0 }}
{{- fail (printf "strata.context must be a number of tokens or 32k / 64k / ..., not %q" $context) }}
{{- end }}
{{- mul $perSlot (.Values.strata.parallel | default 1 | int) }}
{{- else }}
{{- .Values.llamacppContextLength }}
{{- end }}
{{- end }}
