using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TwitterClassification.Models;

namespace TwitterClassification
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "twitter_train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "twitter_test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("content", DataKind.Text, 1)
                }
            });

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);
            SaveModelAsFile(mlContext, model);
            PredictWithModelLoadedFromFile(mlContext);
            Console.ReadLine();
        }

        private static void PredictWithModelLoadedFromFile(MLContext mlContext)
        {
            IEnumerable<TwitterData> sentiments = new[] {
                new TwitterData {  content = "volveré a ir a starbucks"},
                new TwitterData {content ="No creo volver a utilizar esta basura"}
            };
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = mlContext.Model.Load(stream);

            var predFunction = loadedModel.MakePredictionFunction<TwitterData, TwitterPrediction>(mlContext);
            sentiments.AsParallel().ToList().ForEach(item =>
            {
                var resultPrediction = predFunction.Predict(item);
                Console.WriteLine($"Sentiment: {item.content} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");
            });
            Console.WriteLine("=============== End of predictions ===============");
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);

            var predictions = model.Transform(dataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label", "Score", "Probability");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);
            var pipeline = mlContext.Transforms.Text.FeaturizeText("content", "Features")
                .Append(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumn: "Label", featureColumn: "Features"));

            return pipeline.Fit(dataView);
        }
    }
}
