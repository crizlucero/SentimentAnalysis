using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using PLplot;
using System.Diagnostics;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML;
using Microsoft.ML.Core.Data;

namespace TaxiFarePrediction
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);
            _textLoader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Separator = ",",
                HasHeader = true,
                Column = new[] {
                    new TextLoader.Column("VendorId", DataKind.Text, 0),
                    new TextLoader.Column("RateCode", DataKind.Text, 1),
                    new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                    new TextLoader.Column("TripTime", DataKind.R4, 3),
                    new TextLoader.Column("TripDistance", DataKind.R4, 4),
                    new TextLoader.Column("PaymentType", DataKind.Text, 5),
                    new TextLoader.Column("FareAmount", DataKind.R4, 6)
                }
            });
            var model = Train(mlContext, _trainDataPath);
            Evaluate(mlContext, model);
            TestSinglePrediction(mlContext);

            PlotRegressionChart(mlContext, 100, args);
            Console.WriteLine("Press any key to exit...");
            Console.ReadLine();
        }

        public static ITransformer Train(MLContext mlcontext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);
            var pipeline = mlcontext.Transforms.CopyColumns("FareAmount", "Label")
                .Append(mlcontext.Transforms.Categorical.OneHotEncoding("VendorId"))
                .Append(mlcontext.Transforms.Categorical.OneHotEncoding("RateCode"))
                .Append(mlcontext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                .Append(mlcontext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripTime", "TripDistance", "PaymentType"))
                .Append(mlcontext.Regression.Trainers.FastTree());
            var model = pipeline.Fit(dataView);
            SaveModelAsFile(mlcontext, model);
            return model;
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = mlContext.Model.Load(stream);
            var predictionFunction = loadedModel.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(mlContext);
            var taxiTripSample = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0
            };
            var prediction = predictionFunction.Predict(taxiTripSample);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

        private static void PlotRegressionChart(MLContext mlContext, int recordsToRead, string[] args)
        {
            ITransformer trainedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                trainedModel = mlContext.Model.Load(stream);
            var predictionFunction = trainedModel.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            string chartFileName = "TaxiRegressionDistribution";
            using (var pl = new PLStream())
            {
                if (args.Length == 1 && args[0] == "svg")
                {
                    pl.sdev("svg");
                    chartFileName += ".svg";
                    pl.sfnam(chartFileName);
                }
                else
                {
                    pl.sdev("pngcairo");
                    chartFileName += ".png";
                }

                pl.spal0("cmap0_alternate.pal");

                pl.init();

                const int xMinLimit = 0, xMaxLimit = 35, yMinLimit = 0, yMaxLimit = 35;

                pl.env(xMinLimit, xMaxLimit, yMinLimit, yMaxLimit, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes);

                pl.schr(0, 1.25);
                pl.lab("Measured", "Predicted", "Distribution of Taxi Fare Prediction");

                pl.col0(1);

                int totalNumber = recordsToRead;
                var testData = new TaxiTripCsvReader().GetDataFromCsv(_testDataPath, totalNumber).ToList();

                char code = (char)9;

                pl.col0(2);

                double yTotal = 0, xTotal = 0, xyMultiTotal = 0, xSquareTotal = 0;

                for (int i = 0; i < testData.Count; i++)
                {
                    double[] x = new double[1], y = new double[1];

                    var FarePrediction = predictionFunction.Predict(testData[i]);

                    x[0] = testData[i].FareAmount;
                    y[0] = FarePrediction.FareAmount;

                    pl.poin(x, y, code);
                    xTotal += x[0];
                    yTotal += y[0];

                    xyMultiTotal += x[0] * y[0];

                    xSquareTotal += x[0] * x[0];

                    Console.WriteLine($"-------------------------------------------------");
                    Console.WriteLine($"Predicted : {FarePrediction.FareAmount}");
                    Console.WriteLine($"Actual:    {testData[i].FareAmount}");
                    Console.WriteLine($"-------------------------------------------------");
                }

                double minY = yTotal / totalNumber;
                double minX = xTotal / totalNumber;
                double minXY = xyMultiTotal / totalNumber;
                double minXsquare = xSquareTotal / totalNumber;

                double m = ((minX * minY) - minXY) / ((minX * minX) - minXsquare);

                double b = minY - (m * minX);

                double x1 = 1;
                double y1 = (m * x1) + b;

                double x2 = 39;
                double y2 = (m * x2) + b;

                double[] xArray = new double[2], yArray = new double[2];
                xArray[0] = x1;
                yArray[0] = y1;
                xArray[1] = x2;
                yArray[1] = y2;

                pl.col0(4);
                pl.line(yArray, xArray);

                pl.eop();

                pl.gver(out var verText);
                Console.WriteLine($"PLplotVersion {verText}");
            }

            Console.WriteLine("Showing chart...");
            var p = new Process();
            string chartFileNamePath = $@".\{chartFileName}";
            p.StartInfo = new ProcessStartInfo(chartFileNamePath)
            {
                UseShellExecute = true
            };
            p.Start();
        }

        public class TaxiTripCsvReader
        {
            public IEnumerable<TaxiTrip> GetDataFromCsv(string dataLocation, int numMaxRecords) =>
                File.ReadAllLines(dataLocation)
                .Skip(1)
                .Select(x => x.Split(','))
                .Select(x => new TaxiTrip
                {
                    VendorId = x[0],
                    RateCode = x[1],
                    PassengerCount = float.Parse(x[2]),
                    TripTime = float.Parse(x[3]),
                    TripDistance = float.Parse(x[4]),
                    PaymentType = x[5],
                    FareAmount = float.Parse(x[6])
                })
                .Take(numMaxRecords);
        }
    }
}
