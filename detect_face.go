package main

import (
	"flag"
	"fmt"
	"time"

	framework "tensorflow/core/framework"
	pb "tensorflow_serving"
	"go.uber.org/ratelimit"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
)

var (
	serverAddr         = flag.String("server_addr", "127.0.0.1:9000", "The server address in the format of host:port")
	modelName          = flag.String("model_name", "mnist", "TensorFlow model name")
	modelVersion       = flag.Int64("model_version", 1, "TensorFlow model version")
	tls                = flag.Bool("tls", false, "Connection uses TLS if true, else plain TCP")
	caFile             = flag.String("ca_file", "testdata/ca.pem", "The file containning the CA root cert file")
	serverHostOverride = flag.String("server_host_override", "x.test.youtube.com", "The server name use to verify the hostname returned by TLS handshake")
)

func main() {
	flag.Parse()
	var opts []grpc.DialOption
	if *tls {
		var sn string
		if *serverHostOverride != "" {
			sn = *serverHostOverride
		}
		var creds credentials.TransportCredentials
		if *caFile != "" {
			var err error
			creds, err = credentials.NewClientTLSFromFile(*caFile, sn)
			if err != nil {
				grpclog.Fatalf("Failed to create TLS credentials %v", err)
			}
		} else {
			creds = credentials.NewClientTLSFromCert(nil, sn)
		}
		opts = append(opts, grpc.WithTransportCredentials(creds))
	} else {
		opts = append(opts, grpc.WithInsecure())
	}
	conn, err := grpc.Dial(*serverAddr, opts...)
	if err != nil {
		grpclog.Fatalf("fail to dial: %v", err)
	}
	defer conn.Close()
	client := pb.NewPredictionServiceClient(conn)

	rl := ratelimit.New(50)
	go func() {
		for {
			if rl == nil {
				return
			}
			rl.Take()
			go func() {
				inErr := sendReq(client, 50)
				if inErr != nil {
					err = inErr
				}
			}()
		}
	}()
	fmt.Println("warmup")
	time.Sleep(500 * time.Millisecond)

	for i := 100; i < 100000; i += 50 {
		rl = ratelimit.New(i)
		c := time.After(time.Second)
		<-c
		if err != nil {
			fmt.Printf("cannot handle: %d call/sec\n", i)
			break
		} else {
			fmt.Printf("Ok with: %d call/sec\n", i)
		}
	}
	rl = nil
}

func sendReq(client pb.PredictionServiceClient, batchSize int) error {
	pr := newMnistRequest(modelName, modelVersion, batchSize)
	ctx, canc := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer canc()
	_, err := client.Predict(ctx, pr)

	if err != nil {
		fmt.Println(err)
		return err
	} else {
		//fmt.Println("OK")
	}
	//for k, v := range resp.Outputs {
	//
	//	fmt.Printf("tensor: %s, version: %d\n", k, v.VersionNumber)
	//	if v.Dtype != framework.DataType_DT_FLOAT {
	//		fmt.Errorf("wrong type: %s", v.Dtype)
	//	}
	//	printTensorProto(v)
	//}
	return nil
}

func printTP(tp *framework.TensorProto, dim, idx int, indexes []int) int {
	max := tp.TensorShape.Dim[dim]
	isLastDim := dim == len(tp.TensorShape.Dim)-1
	indexes = append(indexes, 0)
	if isLastDim {
		fmt.Printf("%v\n", indexes)
	}
	for i := 0; i < int(max.Size); i++ {
		indexes[dim] = i
		if !isLastDim {
			idx = printTP(tp, dim+1, idx, indexes)
		} else {
			fmt.Printf("%f\n", tp.FloatVal[idx])
			idx++
		}
	}
	return idx
}

func printTensorProto(tp *framework.TensorProto) {
	fmt.Printf("%v\n", tp.TensorShape)
	printTP(tp, 0, 0, nil)
}

func newMnistRequest(modelName *string, modelVersion *int64, batchSize int) *pb.PredictRequest {
	pr := newPredictRequest(*modelName, *modelVersion)
	pr.ModelSpec.SignatureName = "predict_images"

	vals := []float32{}
	const imgSize = 28 * 28
	for n := 0; n < batchSize; n++ {
		for i := 0; i < imgSize; i++ {
			vals = append(vals, 0.5)
		}
	}
	addInput(pr, "images", framework.DataType_DT_FLOAT, vals, []int64{int64(batchSize), imgSize}, nil)
	return pr
}

func newDensePredictRequest(modelName *string, modelVersion *int64) *pb.PredictRequest {
	pr := newPredictRequest(*modelName, *modelVersion)
	addInput(pr, "keys", framework.DataType_DT_INT32, []int32{1, 2, 3}, nil, nil)
	addInput(pr, "features", framework.DataType_DT_FLOAT, []float32{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
		1, 2, 3, 4, 5, 6, 7, 8, 9,
		1, 2, 3, 4, 5, 6, 7, 8, 9,
	}, []int64{3, 9}, nil)
	return pr
}

// Example data:
// 0 5:1 6:1 17:1 21:1 35:1 40:1 53:1 63:1 71:1 73:1 74:1 76:1 80:1 83:1
// 1 5:1 7:1 17:1 22:1 36:1 40:1 51:1 63:1 67:1 73:1 74:1 76:1 81:1 83:1
func newSparsePredictRequest(modelName *string, modelVersion *int64) *pb.PredictRequest {
	pr := newPredictRequest(*modelName, *modelVersion)
	addInput(pr, "keys", framework.DataType_DT_INT32, []int32{1, 2}, nil, nil)
	addInput(pr, "indexs", framework.DataType_DT_INT64, []int64{
		0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5,
		0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11,
		0, 12, 0, 13, 1, 0, 1, 1, 1, 2, 1, 3,
		1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9,
		1, 10, 1, 11, 1, 12, 1, 13,
	}, []int64{28, 2}, nil)
	addInput(pr, "ids", framework.DataType_DT_INT64, []int64{
		5, 6, 17, 21, 35, 40, 53, 63, 71, 73, 74, 76, 80, 83,
		5, 7, 17, 22, 36, 40, 51, 63, 67, 73, 74, 76, 81, 83,
	}, nil, nil)
	addInput(pr, "values", framework.DataType_DT_FLOAT, []float32{
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
	}, nil, nil)
	addInput(pr, "shape", framework.DataType_DT_INT64, []int64{2, 124}, nil, nil)
	return pr
}