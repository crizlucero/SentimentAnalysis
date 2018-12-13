using BikeSharingDemand.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System;
using System.IO;

namespace BikeSharingDemand
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "hour_train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "hour_test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "MLModels");
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);
            var textLoader = BikeSharingTextLoaderFactory.CreateTextLoader(mlContext);
            var trainingDataView = textLoader.Read(_trainDataPath);
            var testDataView = textLoader.Read(_testDataPath);

            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Count", "Label")
                .Append(mlContext.Transforms.Concatenate("Features", "Season", "Year", "Month", "Hour", "Holiday", "Weekday", "Weather", "Temperature", "NormalizedTemperature", "Humidity", "Windspeed"));

            (string name, IEstimator<ITransformer> value)[] regressionLearners =
            {
                ("FastTree", mlContext.Regression.Trainers.FastTree()),
                ("Poisson", mlContext.Regression.Trainers.PoissonRegression()),
                ("SDCA", mlContext.Regression.Trainers.StochasticDualCoordinateAscent()),
                ("FastTreeTweedie", mlContext.Regression.Trainers.FastTreeTweedie()),
            };

            foreach (var learner in regressionLearners)
            {
                Console.WriteLine("=============== Training the current model ===============");
                var trainingPipeline = dataProcessPipeline.Append(learner.value);
                var trainedModel = trainingPipeline.Fit(trainingDataView);

                Console.WriteLine("==== Evaluating Model's accuracy with Test data ====");
                IDataView predictions = trainedModel.Transform(testDataView);
                var metrics = mlContext.Regression.Evaluate(predictions, label: "Count", score: "Score");
                Console.WriteLine($"{learner.value} - {metrics}");

                using (var fs = new FileStream(Path.Combine(Environment.CurrentDirectory, "MLModels", $"{learner.name}Model.zip"), FileMode.Create, FileAccess.Write, FileShare.Write))
                    mlContext.Model.Save(trainedModel, fs);

                Console.WriteLine($"The model is saved to {Path.Combine(Environment.CurrentDirectory, "MLModels", $"{learner.name}Model.zip")}");
            }

            foreach (var learner in regressionLearners)
            {
                ITransformer trainedModel;
                using (var stream = new FileStream(Path.Combine(Environment.CurrentDirectory, "MLModels", $"{learner.name}Model.zip"), FileMode.Open, FileAccess.Read, FileShare.Write))
                    trainedModel = mlContext.Model.Load(stream);
                var predFunction = trainedModel.MakePredictionFunction<DemandObservation, DemandPrediction>(mlContext);

                Console.WriteLine($"=========== Visualize/test 10 predictions for model {learner.name}Model.zip ==============");
                ModelScoringTester.VisualizeSomePredictions(mlContext, learner.name, _testDataPath, predFunction, 10);
            }
            Console.ReadLine();
        }
    }
}
