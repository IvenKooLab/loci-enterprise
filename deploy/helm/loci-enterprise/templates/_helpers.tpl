{{- define "loci-enterprise.fullname" -}}
{{ .Release.Name }}-{{ .Chart.Name }}
{{- end -}}
