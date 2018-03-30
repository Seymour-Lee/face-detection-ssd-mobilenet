package main

import (
	"context"
	"flag"
	// "io/ioutil"
	"log"
	//"reflect"
	// "os"
	// "path/filepath"
	// "image"
	// "image/jpeg"
	"github.com/Comdex/imgo"

	framework "tensorflow/core/framework"
	pb "tensorflow_serving"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
	// "go.uber.org/ratelimit"
	// "golang.org/x/net/context"
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

	imgPath := "/Users/miaozou/Documents/projects/face-detection-ssd-mobilenet/data/tf_wider_train/images/20_Family_Group_Family_Group_20_304.jpg"
	if err != nil {
		log.Fatalln(err)
	}

	// img, err := imgo.DecodeImage("rgb.png")  // 获取 图片 image.Image 对象  
    // if err != nil {  
    //     log.Println(err)  
    // }  
	imgMatrix := imgo.MustRead(imgPath)          // 读取图片RGBA值  
	for i := 0; i < len(imgMatrix); i++	{
		for j := 0; j < len(imgMatrix[i]); j++{
			imgMatrix[i][j] = imgMatrix[i][j][:3]
		}
	}
	img4D := [][][][]uint8 {imgMatrix }
//	var img4D [][][][]uint8
//	img4D = img4D.append(imgMatrix)
	// file, err := os.Open(imgPath)
	// if err != nil {
	// 	log.Fatalln(err)
	// }
	// img, err := jpeg.Decode(file)
	// if err != nil {
	// 	log.Fatalln(err)
	// }
	// log.Println(img.At(10, 10).RGBA())

	tensor, err := tf.NewTensor(img4D)
	if err != nil {
		log.Fatalln("Cannot read image file")
	}
	log.Println(tensor)

	// tensor32, ok := tensor.Value().([][][][]uint8)
	// if !ok {
	// 	log.Fatalln("Cannot type assert tensor value to string")
	// }
	// log.Println(tensor32)
	imgVal := []int32{}
	for i := 0; i < len(img4D); i++	{
		for j := 0; j < len(img4D[i]); j++{
			for k := 0; k < len(img4D[i][j]); k++{
				for l := 0; l < len(img4D[i][j][k]); l++{
					imgVal = append(imgVal, int32(img4D[i][j][k][l]))
					//imgVal = imgVal.append(img4D[i][j][k][l])
				}
			}
		}
	}

	request := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          "mobilenet",
			SignatureName: "serving_default",
			Version: &google_protobuf.Int64Value{
				Value: int64(1),
			},
		},
		Inputs: map[string]*framework.TensorProto{
			"inputs": &framework.TensorProto{
				Dtype: framework.DataType_DT_UINT8,
				TensorShape: &framework.TensorShapeProto{
					Dim: []*framework.TensorShapeProto_Dim{
						&framework.TensorShapeProto_Dim{
							Size: int64(1),
						},
						&framework.TensorShapeProto_Dim{
							Size: int64(1366),
						},
						&framework.TensorShapeProto_Dim{
							Size: int64(1024),
						},
						&framework.TensorShapeProto_Dim{
							Size: int64(3),
						},
					},
				},
				IntVal: imgVal, //This is not right
			},
		},
	}

	


	resp, err := client.Predict(context.Background(), request)
	if err != nil {
		log.Fatalln(err)
	}

	log.Println(resp)

	// rl := ratelimit.New(50)
	// go func() {
	// 	for {
	// 		if rl == nil {
	// 			return
	// 		}
	// 		rl.Take()
	// 		go func() {
	// 			inErr := sendReq(client, 50)
	// 			if inErr != nil {
	// 				err = inErr
	// 			}
	// 		}()
	// 	}
	// }()
	// fmt.Println("warmup")
	// time.Sleep(500 * time.Millisecond)

	// for i := 100; i < 100000; i += 50 {
	// 	rl = ratelimit.New(i)
	// 	c := time.After(time.Second)
	// 	<-c
	// 	if err != nil {
	// 		fmt.Printf("cannot handle: %d call/sec\n", i)
	// 		break
	// 	} else {
	// 		fmt.Printf("Ok with: %d call/sec\n", i)
	// 	}
	// }
	// rl = nil
}

// func sendReq(client pb.PredictionServiceClient, batchSize int) error {
// 	pr := newMnistRequest(modelName, modelVersion, batchSize)
// 	ctx, canc := context.WithTimeout(context.Background(), 20*time.Millisecond)
// 	defer canc()
// 	_, err := client.Predict(ctx, pr)

// 	if err != nil {
// 		fmt.Println(err)
// 		return err
// 	} else {
// 		//fmt.Println("OK")
// 	}
// 	//for k, v := range resp.Outputs {
// 	//
// 	//	fmt.Printf("tensor: %s, version: %d\n", k, v.VersionNumber)
// 	//	if v.Dtype != framework.DataType_DT_FLOAT {
// 	//		fmt.Errorf("wrong type: %s", v.Dtype)
// 	//	}
// 	//	printTensorProto(v)
// 	//}
// 	return nil
// }

// func printTP(tp *framework.TensorProto, dim, idx int, indexes []int) int {
// 	max := tp.TensorShape.Dim[dim]
// 	isLastDim := dim == len(tp.TensorShape.Dim)-1
// 	indexes = append(indexes, 0)
// 	if isLastDim {
// 		fmt.Printf("%v\n", indexes)
// 	}
// 	for i := 0; i < int(max.Size); i++ {
// 		indexes[dim] = i
// 		if !isLastDim {
// 			idx = printTP(tp, dim+1, idx, indexes)
// 		} else {
// 			fmt.Printf("%f\n", tp.FloatVal[idx])
// 			idx++
// 		}
// 	}
// 	return idx
// }

// func printTensorProto(tp *framework.TensorProto) {
// 	fmt.Printf("%v\n", tp.TensorShape)
// 	printTP(tp, 0, 0, nil)
// }

// func newMnistRequest(modelName *string, modelVersion *int64, batchSize int) *pb.PredictRequest {
// 	pr := newPredictRequest(*modelName, *modelVersion)
// 	pr.ModelSpec.SignatureName = "predict_images"

// 	vals := []float32{}
// 	const imgSize = 28 * 28
// 	for n := 0; n < batchSize; n++ {
// 		for i := 0; i < imgSize; i++ {
// 			vals = append(vals, 0.5)
// 		}
// 	}
// 	addInput(pr, "images", framework.DataType_DT_FLOAT, vals, []int64{int64(batchSize), imgSize}, nil)
// 	return pr
// }
