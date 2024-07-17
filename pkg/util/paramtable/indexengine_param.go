package paramtable

import "strings"

type indexEngineConfig struct {
	indexParam ParamGroup `refreshable:"true"`
}

const (
	buildStage  = "build"
	loadStage   = "load"
	searchStage = "search"
)

func (p *indexEngineConfig) init(base *BaseTable) {
	p.indexParam = ParamGroup{
		KeyPrefix: "knowhere.",
		Version:   "2.5.0",
	}
	p.indexParam.Init(base.mgr)
	val := p.indexParam.GetValue()
	print(val)
}

func (p *indexEngineConfig) getIndexParam(indexType string, stage string) map[string]string {
	matchedParam := make(map[string]string)

	params := p.indexParam.GetValue()
	prefix := indexType + "." + stage + "."

	for k, v := range params {
		if strings.HasPrefix(k, prefix) {
			matchedParam[strings.TrimPrefix(k, prefix)] = v
		}
	}

	return matchedParam
}
