using System;
using System.Threading.Tasks;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Core.Data;
using System.Diagnostics;
using GitHubLabeler.DataStructures;
using Microsoft.Extensions.Configuration;

namespace GitHubLabeler
{
    internal static class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "corefx-issues-train.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "MLModels", "GitHubLabelerModel.zip");

        public enum MyTrainerStrategy : int
        {
            SdcaMultiClassTrainer = 1,
            OVAAveragedPreceptronTrainer = 2
        };

        public static IConfiguration Configuration { get; set; }
        private static async Task Main(string[] args)
        {
            SetupAppConfiguration();

            BuildAndTrainModel(_trainDataPath, _modelPath, MyTrainerStrategy.OVAAveragedPreceptronTrainer);

            TestSingleLabelPrediction(_modelPath);

            await PredictLabelsAndUpdateGitHub(_modelPath);

            //Common.ConsoleHelper.ConsolePressAnyKey();
        }

        private static async Task PredictLabelsAndUpdateGitHub(string modelPath)
        {
            var token = Configuration["GitHubToken"];
            var repoOwner = Configuration["GitHubRepoOwner"];
            var repoName = Configuration["GitHubRepoName"];

            if (string.IsNullOrEmpty(token) ||
                string.IsNullOrEmpty(repoOwner) ||
                string.IsNullOrEmpty(repoName))
            {
                Console.Error.WriteLine();
                Console.Error.WriteLine("Error: please configure the credetnials in the appsettings.json file");
                Console.ReadLine();
                return;
            }

            var labeler = new Labeler(_modelPath, repoOwner, repoName, token);
            await labeler.LabelAllNewIssuesInGitHubRepo();

            Console.WriteLine("Labeling completed");
            Console.ReadLine();
        }

        private static void TestSingleLabelPrediction(string modelPath) =>
            new Labeler(modelPath).TestPredictionForSingleIssue();

        private static void BuildAndTrainModel(string trainDataPath, string modelPath, MyTrainerStrategy selectedStrategy)
        {
            var mlContext = new MLContext(seed: 0);

            TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]{
                    new TextLoader.Column("ID", DataKind.Text, 0),
                    new TextLoader.Column("Area", DataKind.Text, 1),
                    new TextLoader.Column("Title", DataKind.Text, 2),
                    new TextLoader.Column("Description", DataKind.Text, 3)
                }
            });

            var trainingDataView = textLoader.Read(_trainDataPath);

            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Area", "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("Title", "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText("Description", "DescriptionFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mlContext);

            IEstimator<ITransformer> trainer = null;
            switch (selectedStrategy)
            {
                case MyTrainerStrategy.SdcaMultiClassTrainer:
                    trainer = mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label,
                        DefaultColumnNames.Features);
                    break;
                case MyTrainerStrategy.OVAAveragedPreceptronTrainer:
                    var averagedPerceptronBinaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron(DefaultColumnNames.Label,
                        DefaultColumnNames.Features,
                        numIterations: 10);
                    trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer);
                    break;
                default: break;
            }

            var trainingPipeline = dataProcessPipeline.Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");

            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeline, numFolds: 6, labelColumn: "Label");

            Console.WriteLine("=============== Training the model ===============");

            var watch = Stopwatch.StartNew();

            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();

            long elapsedMS = watch.ElapsedMilliseconds;

            GitHubIssue issue = new GitHubIssue
            {
                ID = "Any-ID",
                Title = "WebSockets communicationn is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var predFunction = trainedModel.MakePredictionFunction<GitHubIssue, GitHubIssuePrediction>(mlContext);

            var prediction = predFunction.Predict(issue);
            Console.WriteLine($"========= Single Prediction just-trained-model - Result: {prediction.Area} ===========");

            Console.WriteLine($"========= Saving the model to a file ==========");
            using (var fs = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);
        }

        private static void SetupAppConfiguration()
        {
            var builder = new ConfigurationBuilder()
               .SetBasePath(Directory.GetCurrentDirectory())
               .AddJsonFile("appsettings.json");
            Configuration = builder.Build();
        }
    }
}
