// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0

package delegator

/*
#cgo pkg-config: milvus_core

#include <stdlib.h>
#include "common/type_c.h"
#include "indexbuilder/index_c.h"
*/
import "C"

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

var errHeadIndexSearchUnsupported = errors.New("global head index search unsupported")

var newHeadIndexSearcherFromPath = func(headIndexPath string) (headIndexSearcher, error) {
	searcher := newHeadIndexSearcher(headIndexPath)
	if searcher == nil {
		return nil, nil
	}
	cardinalSearcher := searcher.(*cardinalHeadIndexSearcher)
	if err := cardinalSearcher.load(); err != nil {
		return nil, err
	}
	return searcher, nil
}

type cardinalHeadIndexSearcher struct {
	headIndexPath string
	storageConfig cStorageConfig

	mut sync.Mutex
	ptr C.CCardinalHeadIndex
}

type cStorageConfig struct {
	address          string
	bucketName       string
	accessKeyID      string
	secretAccessKey  string
	rootPath         string
	storageType      string
	cloudProvider    string
	iamEndpoint      string
	region           string
	sslCACert        string
	gcpCredential    string
	tlsMinVersion    string
	useSSL           bool
	useIAM           bool
	useVirtualHost   bool
	requestTimeoutMs int64
	maxConnections   uint32
	useCRC32C        bool
}

func newHeadIndexSearcher(headIndexPath string) headIndexSearcher {
	if headIndexPath == "" {
		return nil
	}
	searcher := &cardinalHeadIndexSearcher{
		headIndexPath: headIndexPath,
		storageConfig: newHeadIndexStorageConfig(),
	}
	runtime.SetFinalizer(searcher, func(searcher *cardinalHeadIndexSearcher) {
		searcher.close()
	})
	return searcher
}

func newHeadIndexStorageConfig() cStorageConfig {
	params := paramtable.Get()
	if params.CommonCfg.StorageType.GetValue() == "local" {
		return cStorageConfig{
			rootPath:    params.LocalStorageCfg.Path.GetValue(),
			storageType: params.CommonCfg.StorageType.GetValue(),
		}
	}
	return cStorageConfig{
		address:          params.MinioCfg.Address.GetValue(),
		bucketName:       params.MinioCfg.BucketName.GetValue(),
		accessKeyID:      params.MinioCfg.AccessKeyID.GetValue(),
		secretAccessKey:  params.MinioCfg.SecretAccessKey.GetValue(),
		rootPath:         params.MinioCfg.RootPath.GetValue(),
		storageType:      params.CommonCfg.StorageType.GetValue(),
		cloudProvider:    params.MinioCfg.CloudProvider.GetValue(),
		iamEndpoint:      params.MinioCfg.IAMEndpoint.GetValue(),
		region:           params.MinioCfg.Region.GetValue(),
		sslCACert:        params.MinioCfg.SslCACert.GetValue(),
		gcpCredential:    params.MinioCfg.GcpCredentialJSON.GetValue(),
		tlsMinVersion:    params.MinioCfg.SslTLSMinVersion.GetValue(),
		useSSL:           params.MinioCfg.UseSSL.GetAsBool(),
		useIAM:           params.MinioCfg.UseIAM.GetAsBool(),
		useVirtualHost:   params.MinioCfg.UseVirtualHost.GetAsBool(),
		requestTimeoutMs: params.MinioCfg.RequestTimeoutMs.GetAsInt64(),
		maxConnections:   uint32(params.MinioCfg.MaxConnections.GetAsInt()),
		useCRC32C:        params.MinioCfg.UseCRC32C.GetAsBool(),
	}
}

func (s *cardinalHeadIndexSearcher) Search(ctx context.Context, req *internalpb.SearchRequest, topK int64) ([][]int64, error) {
	_ = ctx
	vectors, dim, ok, err := parseHeadIndexSearchFloatVectors(req)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, errHeadIndexSearchUnsupported
	}
	if topK <= 0 {
		topK = 1
	}
	if err := s.load(); err != nil {
		return nil, err
	}

	nq := int64(len(vectors)) / dim
	ids := make([]int64, nq*topK)
	status := C.SearchCardinalHeadIndex(
		s.ptr,
		(*C.float)(unsafe.Pointer(&vectors[0])),
		C.int64_t(nq),
		C.int64_t(dim),
		C.int64_t(topK),
		(*C.int64_t)(unsafe.Pointer(&ids[0])),
	)
	if err := handleHeadIndexCStatus(&status, "search global head index"); err != nil {
		return nil, err
	}

	result := make([][]int64, nq)
	for q := int64(0); q < nq; q++ {
		for k := int64(0); k < topK; k++ {
			id := ids[q*topK+k]
			if id >= 0 {
				result[q] = append(result[q], id)
			}
		}
	}
	return result, nil
}

func (s *cardinalHeadIndexSearcher) load() error {
	s.mut.Lock()
	defer s.mut.Unlock()
	if s.ptr != nil {
		return nil
	}

	cPath := C.CString(s.headIndexPath)
	defer C.free(unsafe.Pointer(cPath))
	cConfig, cleanup := s.storageConfig.toC()
	defer cleanup()

	var ptr C.CCardinalHeadIndex
	status := C.LoadCardinalHeadIndex(&ptr, cConfig, cPath)
	if err := handleHeadIndexCStatus(&status, "load global head index"); err != nil {
		return err
	}
	s.ptr = ptr
	return nil
}

func (s *cardinalHeadIndexSearcher) close() {
	s.mut.Lock()
	defer s.mut.Unlock()
	if s.ptr == nil {
		return
	}
	status := C.DeleteCardinalHeadIndex(s.ptr)
	if err := handleHeadIndexCStatus(&status, "delete global head index"); err != nil {
		log.Warn(fmt.Sprintf("failed to delete global head index: %v", err))
	}
	s.ptr = nil
}

func parseHeadIndexSearchFloatVectors(req *internalpb.SearchRequest) ([]float32, int64, bool, error) {
	if len(req.GetPlaceholderGroup()) == 0 {
		return nil, 0, false, nil
	}
	group := &commonpb.PlaceholderGroup{}
	if err := proto.Unmarshal(req.GetPlaceholderGroup(), group); err != nil {
		return nil, 0, false, merr.WrapErrParameterInvalidMsg("invalid search vector placeholder: %v", err)
	}
	if len(group.GetPlaceholders()) == 0 {
		return nil, 0, false, nil
	}

	placeholder := group.GetPlaceholders()[0]
	if placeholder.GetType() != commonpb.PlaceholderType_FloatVector {
		return nil, 0, false, nil
	}
	values := placeholder.GetValues()
	if len(values) == 0 {
		return nil, 0, false, nil
	}
	if len(values[0]) == 0 || len(values[0])%4 != 0 {
		return nil, 0, false, merr.WrapErrParameterInvalidMsg("invalid float vector placeholder length: %d", len(values[0]))
	}

	dim := int64(len(values[0]) / 4)
	vectors := make([]float32, 0, len(values)*int(dim))
	for _, value := range values {
		if len(value)%4 != 0 {
			return nil, 0, false, merr.WrapErrParameterInvalidMsg("invalid float vector placeholder length: %d", len(value))
		}
		if int64(len(value)/4) != dim {
			return nil, 0, false, merr.WrapErrParameterInvalidMsg("inconsistent float vector placeholder dim")
		}
		for offset := 0; offset < len(value); offset += 4 {
			vectors = append(vectors, math.Float32frombits(binary.LittleEndian.Uint32(value[offset:])))
		}
	}
	return vectors, dim, true, nil
}

func handleHeadIndexCStatus(status *C.CStatus, extraInfo string) error {
	if status.error_code == 0 {
		return nil
	}
	errorCode := int32(status.error_code)
	errorMsg := C.GoString(status.error_msg)
	defer C.free(unsafe.Pointer(status.error_msg))
	log.Warn(fmt.Sprintf("%s, C Runtime Exception: %s", extraInfo, errorMsg))
	return merr.SegcoreError(errorCode, errorMsg)
}

func (cfg cStorageConfig) toC() (C.CStorageConfig, func()) {
	cAddress := C.CString(cfg.address)
	cBucketName := C.CString(cfg.bucketName)
	cAccessKeyID := C.CString(cfg.accessKeyID)
	cSecretAccessKey := C.CString(cfg.secretAccessKey)
	cRootPath := C.CString(cfg.rootPath)
	cStorageType := C.CString(cfg.storageType)
	cCloudProvider := C.CString(cfg.cloudProvider)
	cIAMEndpoint := C.CString(cfg.iamEndpoint)
	cRegion := C.CString(cfg.region)
	cSSLCACert := C.CString(cfg.sslCACert)
	cGCPCredential := C.CString(cfg.gcpCredential)
	cTLSMinVersion := C.CString(cfg.tlsMinVersion)

	cleanup := func() {
		C.free(unsafe.Pointer(cAddress))
		C.free(unsafe.Pointer(cBucketName))
		C.free(unsafe.Pointer(cAccessKeyID))
		C.free(unsafe.Pointer(cSecretAccessKey))
		C.free(unsafe.Pointer(cRootPath))
		C.free(unsafe.Pointer(cStorageType))
		C.free(unsafe.Pointer(cCloudProvider))
		C.free(unsafe.Pointer(cIAMEndpoint))
		C.free(unsafe.Pointer(cRegion))
		C.free(unsafe.Pointer(cSSLCACert))
		C.free(unsafe.Pointer(cGCPCredential))
		C.free(unsafe.Pointer(cTLSMinVersion))
	}

	return C.CStorageConfig{
		address:             cAddress,
		bucket_name:         cBucketName,
		access_key_id:       cAccessKeyID,
		access_key_value:    cSecretAccessKey,
		root_path:           cRootPath,
		storage_type:        cStorageType,
		cloud_provider:      cCloudProvider,
		iam_endpoint:        cIAMEndpoint,
		region:              cRegion,
		sslCACert:           cSSLCACert,
		gcp_credential_json: cGCPCredential,
		tls_min_version:     cTLSMinVersion,
		useSSL:              C.bool(cfg.useSSL),
		useIAM:              C.bool(cfg.useIAM),
		useVirtualHost:      C.bool(cfg.useVirtualHost),
		requestTimeoutMs:    C.int64_t(cfg.requestTimeoutMs),
		max_connections:     C.uint32_t(cfg.maxConnections),
		use_crc32c_checksum: C.bool(cfg.useCRC32C),
	}, cleanup
}
