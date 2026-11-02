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
The local LLM server this release runs (llm.engine): llamacpp or strata.
Anything else fails the render rather than quietly running neither.
*/}}
{{/*
LLM_SLOTS: llm.slots, else the running engine's parallel. core uses it only
when the server's /props has no total_slots.
*/}}
{{- define "dcb.llmSlots" -}}
{{- if .Values.llm.slots }}
{{- .Values.llm.slots | toString }}
{{- else if eq (include "dcb.llmEngine" .) "strata" }}
{{- .Values.strata.parallel | toString }}
{{- else }}
{{- .Values.llamacpp.parallel | toString }}
{{- end }}
{{- end }}

{{/*
llamacpp.cacheRamMiB as llama.cpp's --cache-ram, or "" to leave its default.
Empty and null (an override that deletes the default) both mean unset; 0
(off) and -1 (no limit) are real values, so this is not a plain truthiness
test. Anything that is not a whole number fails the render.
*/}}
{{- define "dcb.cacheRamMiB" -}}
{{- $v := .Values.llamacpp.cacheRamMiB }}
{{- if kindIs "string" $v }}
{{- if $v }}
{{- if not (regexMatch "^-?[0-9]+$" $v) }}
{{- fail (printf "llamacpp.cacheRamMiB must be a whole number of MiB (0 = off, -1 = no limit), not %q" $v) }}
{{- end }}
{{- $v }}
{{- end }}
{{- else if or (kindIs "float64" $v) (kindIs "int" $v) (kindIs "int64" $v) }}
{{- if ne (float64 $v) (floor $v) }}
{{- fail (printf "llamacpp.cacheRamMiB must be a whole number of MiB, not %v" $v) }}
{{- end }}
{{- int64 $v }}
{{- else if not (kindIs "invalid" $v) }}
{{- fail (printf "llamacpp.cacheRamMiB must be a whole number of MiB, not %v" $v) }}
{{- end }}
{{- end }}

{{/*
With engine=strata, core's LLM_API_KEY is Strata's own key (core.yaml). A
role's or side job's own key with no host of its own would replace it for
that job (llm_roles prefers the job's key), and Strata refuses every other
key: say a persona.apiKey left over from a remote persona, and every chat
reply fails. So such a key fails the render. A side job with no host of
its own uses the brain's, so it is fine when brain.llmHost is set. The same holds for a server
outside the chart named by llm.host, but there the operator chose the key.
*/}}
{{- define "dcb.checkStrataKeys" -}}
{{- if eq (include "dcb.llmEngine" .) "strata" }}
{{- /* name: (the job, the host it falls back to: side jobs use the brain's) */}}
{{- $jobs := dict
  "persona" (list .Values.persona "")
  "brain" (list .Values.brain "")
  "research" (list .Values.research .Values.brain.llmHost)
  "diffusion.promptRewrite" (list .Values.diffusion.promptRewrite .Values.brain.llmHost)
  "diffusion.review" (list .Values.diffusion.review .Values.brain.llmHost)
}}
{{- range $name, $pair := $jobs }}
{{- $job := index $pair 0 }}
{{- if and $job.apiKey (not (or $job.llmHost (index $pair 1))) }}
{{- fail (printf "%s.apiKey is set but %s.llmHost is not: with llm.engine strata it would replace Strata's own key for %s, which Strata refuses. Set %s.llmHost to the server the key is for, or remove the key." $name $name $name $name) }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}

{{- define "dcb.llmEngine" -}}
{{- $engine := .Values.llm.engine | default "llamacpp" }}
{{- if not (has $engine (list "llamacpp" "strata")) }}
{{- fail (printf "llm.engine must be llamacpp or strata, not %q" $engine) }}
{{- end }}
{{- $engine }}
{{- end }}

{{/*
core's LLM_HOST: llm.host when set, else the in-cluster service of the
selected engine. An explicit llm.host is kept as-is, so it must be changed
too when switching engines.
*/}}
{{- define "dcb.llmHost" -}}
{{- if .Values.llm.host }}
{{- .Values.llm.host }}
{{- else if eq (include "dcb.llmEngine" .) "strata" }}
{{- printf "http://%s-strata-service.%s.svc.cluster.local:%v" .Release.Name .Release.Namespace .Values.strata.port }}
{{- else }}
{{- printf "http://%s-llamacpp-service.%s.svc.cluster.local:%v" .Release.Name .Release.Namespace .Values.llamacpp.port }}
{{- end }}
{{- end }}

{{/*
The Secret holding Strata's API key (key STRATA_API_KEY): read by the Strata
pod (API_KEY) and by core (LLM_API_KEY), so the two can never disagree. Strata
always runs with a key; without one of ours it would make its own, which
core would never see, so the render fails instead.
*/}}
{{- define "dcb.strataSecretName" -}}
{{- if .Values.strata.existingSecret }}
{{- .Values.strata.existingSecret }}
{{- else if .Values.strata.apiKey }}
{{- printf "%s-strata-secret" .Release.Name }}
{{- else }}
{{- fail "llm.engine=strata needs strata.apiKey or strata.existingSecret (key STRATA_API_KEY): Strata will not serve without an API key" }}
{{- end }}
{{- end }}

{{/*
A positive whole number, as digits; fails the render otherwise. For counts a
template multiplies: sprig's mul turns anything it cannot parse ("40k") into
0, which would quietly become llama.cpp's "use the model's own context".
Values numbers arrive as float64 (so 1000000 is 1e+06 as a string), hence the
numeric branch. Usage: include "dcb.wholeNumber" (list "key.path" value)
*/}}
{{- define "dcb.wholeNumber" -}}
{{- $name := index . 0 }}
{{- $v := index . 1 }}
{{- $ok := false }}
{{- if kindIs "string" $v }}
{{- $ok = regexMatch "^[1-9][0-9]*$" $v }}
{{- else if or (kindIs "float64" $v) (kindIs "int" $v) (kindIs "int64" $v) }}
{{- $ok = and (gt (float64 $v) 0.0) (eq (float64 $v) (floor $v)) }}
{{- end }}
{{- if not $ok }}
{{- fail (printf "%s must be a positive whole number, not %v" $name $v) }}
{{- end }}
{{- int64 $v }}
{{- end }}
