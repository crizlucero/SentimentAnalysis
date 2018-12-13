using ImageClassification.ImageDataStructures;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ImageClassification.ModelScorer
{
    public class TFModelScorer
    {
        readonly string dataLocation;
        readonly string imagesFolder;
        readonly string modelLocation;
        readonly string labelsLocation;
        readonly MLContext mlContext;

        public TFModelScorer(string dataLocation, string imagesFolder, string modelLocation, string labelsLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.labelsLocation = labelsLocation;
            mlContext = new MLContext();
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
            public const float mean = 117;
            public const bool chanelsLast = true;
        }

        public struct InceptionSettings
        {
            public const string inputTensorName = "input";
            public const string outputTensorName = "softmax2";
        }

        public void Score()
        {
            var model = LoadModel(dataLocation, imagesFolder, modelLocation);
            var predictions = PredictDataUsingModel(dataLocation, imagesFolder, labelsLocation, model).ToArray();
        }

        PredictionFunction<ImageNetData, ImageNetPrediction> LoadModel(string dataLocation, string imagesFolder, string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {dataLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight}), image mean: {ImageNetSettings.mean}");

            var loader = new TextLoader(mlContext,
                new TextLoader.Arguments
                {
                    Column = new[]
                    {
                        new TextLoader.Column("ImagePath", DataKind.Text, 0)
                    }
                });

            var data = loader.Read(new MultiFileSource(dataLocation));

            var pipeline = mlContext.Transforms.LoadImages(imageFolder: imagesFolder, columns: ("ImagePath", "ImageReal"))
                .Append(mlContext.Transforms.Resize("ImageReal", "ImageReal", ImageNetSettings.imageHeight, ImageNetSettings.imageWidth))
                .Append(mlContext.Transforms.ExtractPixels(new[] { new ImagePixelExtractorTransform.ColumnInfo("ImageReal", "input", interleave: ImageNetSettings.chanelsLast, offset: ImageNetSettings.mean) }))
                .Append(mlContext.Transforms.ScoreTensorFlowModel(modelLocation, new[] { "input" }, new[] { "softmax2" }));

            var model = pipeline.Fit(data);
            var predictionFunction = model.MakePredictionFunction<ImageNetData, ImageNetPrediction>(mlContext);

            return predictionFunction;
        }

        protected IEnumerable<ImageNetData> PredictDataUsingModel(string testLocation, string imageFolder, string labelsLocation, PredictionFunction<ImageNetData, ImageNetPrediction> model)
        {
            Console.WriteLine("Classificate images");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine($"Labels file: {labelsLocation}");

            var labels = ModelHelpers.ReadLabels(labelsLocation);

            var testData = ImageNetData.ReadFromCsv(testLocation, imageFolder);

            foreach (var sample in testData)
            {
                var probs = model.Predict(sample).PredictedLabels;
                var imageData = new ImageNetDataProbability()
                {
                    ImagePath = sample.ImagePath,
                    Label = sample.Label
                };
                (imageData.PredictedLabel, imageData.Probability) = ModelHelpers.GetBestLabel(labels, probs);
                imageData.ConsoleWrite();
                yield return imageData;
            }
        }
    }
}