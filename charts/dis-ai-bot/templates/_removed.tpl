{{/*
Values keys renamed or removed in v3 (docs/upgrading-to-v3.md). No aliases:
a stale values file must not render with defaults in place of its settings
(a remote brain would quietly lose its key, for example), so any old key
fails the render, naming each one and what replaced it. configmap.yaml
includes this, so every render runs it. core does the same for env vars
(core/classes/settings_check.py). Each phase adds the keys it moves.
*/}}
{{- define "dcb.failOnRemoved" -}}
{{- $found := list }}
{{- $top := dict
  "llmEngine" "llm.engine"
  "model" "llm.model"
  "llm_host" "llm.host"
  "llm_pass" "llm.apiKey"
  "reasoningEffort" "persona.reasoningEffort"
  "llmMaxTurns" "persona.maxTurns"
  "llmCallTimeout" "persona.callTimeout"
  "llamacppContextLength" "llamacpp.contextPerSlot (the per-slot context; the total is it x llamacpp.parallel)"
  "message_history" "history.limit"
  "worker_count" "queue.workers"
  "queue_max_size" "queue.maxSize"
  "openaiApiKey" "contentGuard.apiKey"
  "openaiBaseUrl" "contentGuard.baseUrl"
  "moderationModel" "contentGuard.model"
  "guardMaxRetries" "contentGuard.maxRetries"
  "guardRetryBackoff" "contentGuard.retryBackoff"
  "moderationCacheTtl" "contentGuard.cacheTtlSeconds"
}}
{{- range $old, $new := $top }}
{{- if hasKey $.Values $old }}
{{- $found = append $found (printf "%s -> %s" $old $new) }}
{{- end }}
{{- end }}
{{- $sandbox := dict
  "model" "brain.model"
  "llmHost" "brain.llmHost"
  "apiKey" "brain.apiKey"
  "requestTimeout" "brain.callTimeout"
  "maxRetries" "removed (the brain's client retries twice; brain.callTimeout bounds one call)"
}}
{{- if kindIs "map" .Values.sandbox }}
{{- range $old, $new := $sandbox }}
{{- if hasKey $.Values.sandbox $old }}
{{- $found = append $found (printf "sandbox.%s -> %s" $old $new) }}
{{- end }}
{{- end }}
{{- end }}
{{- if $found }}
{{- fail (printf "\nThese values keys were renamed or removed in v3 (see docs/upgrading-to-v3.md):\n  %s\n" (join "\n  " (sortAlpha $found))) }}
{{- end }}
{{- end }}
