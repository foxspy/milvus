package paramtable

import "testing"

func TestIndexEngineConfig_Init(t *testing.T) {
	params := ComponentParam{}
	params.Init(NewBaseTable(SkipRemote(true)))

	cfg := &params.indexEngineConfig
	print(cfg)
}

func TestIndexEngineConfig_Get(t *testing.T) {
	params := ComponentParam{}
	params.Init(NewBaseTable(SkipRemote(true)))

	cfg := &params.indexEngineConfig
	diskANNbuild := cfg.getIndexParam("DISKANN", buildStage)
	print(diskANNbuild)
}
