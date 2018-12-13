using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using SpamDetection.MLDataStructures;
using System;
using System.ComponentModel.Composition;
using System.IO;
using System.Linq;

namespace SpamDetection
{
    class Program
    {
        static string TrainDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "spamfolder", "SMSSpamCollection");
        static void Main(string[] args)
        {
            if (!File.Exists(TrainDataPath))
            {
                Utils.DocumentsUtil.DownloadFile();
            }
            var mlcontext = new MLContext();

            var reader = new TextLoader(mlcontext, new TextLoader.Arguments
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Text, 0),
                    new TextLoader.Column("Message", DataKind.Text, 1)
                }
            });

            var data = reader.Read(new MultiFileSource(TrainDataPath));

            var estimator = mlcontext.Transforms.CustomMapping<MyInput, MyOutput>(MyLambda.MyAction, "My Lambda")
                .Append(mlcontext.Transforms.Text.FeaturizeText("Message", "Features"))
                .Append(mlcontext.BinaryClassification.Trainers.StochasticDualCoordinateAscent());

            var cvResults = mlcontext.BinaryClassification.CrossValidate(data, estimator, numFolds: 5);
            var aucs = cvResults.Select(r => r.metrics.Auc);
            Console.WriteLine($"The AUC is {aucs.Average()}");

            var model = estimator.Fit(data);

            var inPipe = new TransformerChain<ITransformer>(model.Take(model.Count() - 1).ToArray());
            var lastTranformer = new BinaryPredictionTransformer<IPredictorProducing<float>>(mlcontext, model.LastTransformer.Model, inPipe.GetOutputSchema(data.Schema), model.LastTransformer.FeatureColumn, threshold: 0.15f, thresholdColumn: DefaultColumnNames.Probability);

            ITransformer[] parts = model.ToArray();
            parts[parts.Length - 1] = lastTranformer;
            var newModel = new TransformerChain<ITransformer>(parts);

            var predictor = newModel.MakePredictionFunction<SpamInput, SpamPrediction>(mlcontext);

            ClassifyMessage(predictor, "That's a great idea. It should work.");
            ClassifyMessage(predictor, "free medicine winner! congratulations");
            ClassifyMessage(predictor, "Yes we should meet over the weekend!");
            ClassifyMessage(predictor, "you win pills and free entry vouchers");
        }
        class MyInput
        {
            public string Label { get; set; }
        }

        class MyOutput
        {
            public bool Label { get; set; }
        }

        class MyLambda
        {
            [Export("MyLambda")]
            public ITransformer MyTransformer => ML.Transforms.CustomMappingTransformer<MyInput, MyOutput>(MyAction, "MyLambda");

            [Import]
            public MLContext ML { get; set; }

            public static void MyAction(MyInput input, MyOutput output) =>
                output.Label = input.Label == "spam";
        }

        public static void ClassifyMessage(PredictionFunction<SpamInput, SpamPrediction> predictor, string message)
        {
            var input = new SpamInput { Message = message };
            var prediction = predictor.Predict(input);

            Console.WriteLine($"The message {input.Message} is {(prediction.IsSpam ? "spam" : "not spam")}");
        }
    }
}
