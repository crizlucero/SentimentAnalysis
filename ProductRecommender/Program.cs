using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;

namespace ProductRecommender
{
    class Program
    {
        private static string _trainingDataLocation = Path.Combine(Environment.CurrentDirectory, "Data", "Amazon0302.txt");
        private static string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model.zip");
        static void Main(string[] args)
        {
            var ctx = new MLContext();

            var reader = ctx.Data.TextReader(new TextLoader.Arguments
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]{
                new TextLoader.Column("Label", DataKind.R4,0),
                new TextLoader.Column("ProductID", DataKind.U4, new[]{
                    new TextLoader.Range(0) },new KeyRange(0,262110)),
                new TextLoader.Column("CoPurchaseProductID", DataKind.U4, new[]{
                    new TextLoader.Range(1) },new KeyRange(0,262110))
                }
            });

            var trainData = reader.Read(new MultiFileSource(_trainingDataLocation));

           /* var est = ctx.Recommendation().Trainers.MatrixFactorization("ProductID", "CoPurchaseProductID", labelColumn: "Label",
                                     advancedSettings: s =>
                                     {
                                         s.LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass;
                                         s.Alpha = 0.01;
                                         s.Lambda = 0.025;
                                         // For better results use the following parameters
                                         //s.K = 100;
                                         //s.C = 0.00001;
                                     });
            var model = est.Fit(trainData);


            //STEP 6: Create prediction engine and predict the score for Product 63 being co-purchased with Product 3.
            //        The higher the score the higher the probability for this particular productID being co-purchased 
            var predictionengine = model.MakePredictionFunction<ProductEntry, Copurchase_prediction>(ctx);
            var prediction = predictionengine.Predict(
                new ProductEntry()
                {
                    ProductID = 3,
                    CoPurchaseProductID = 63
                });*/
        }

        public class Copurchase_prediction
        {
            public float Score { get; set; }
        }

        public class ProductEntry
        {
            [KeyType(Contiguous = true, Count = 262111, Min = 0)]
            public uint ProductID { get; set; }

            [KeyType(Contiguous = true, Count = 262111, Min = 0)]
            public uint CoPurchaseProductID { get; set; }
        }
    }
}
