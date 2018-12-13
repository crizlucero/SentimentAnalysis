using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BikeSharingDemand.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;

namespace BikeSharingDemand
{
    public static class ModelScoringTester
    {
        public static void VisualizeSomePredictions(MLContext mlContext, string modelName, string testDataLocation, PredictionFunction<DemandObservation, DemandPrediction> predFunction, int numberOfPredictions)
        {
            var testData = ReadSampleDataFromCsvFile(testDataLocation, numberOfPredictions);
            for (int i = 0; i < numberOfPredictions; i++)
            {
                var resultPrediction = predFunction.Predict(testData[i]);
                Console.WriteLine($"{resultPrediction.PredictedCount} - {testData[i].Count}");
            }
        }

        private static List<DemandObservation> ReadSampleDataFromCsvFile(string DataLocation, int numberOfPredictionsToRead) =>
         File.ReadLines(DataLocation)
            .Skip(1)
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .Select(x => x.Split(','))
            .Select(x => new DemandObservation
            {
                Season = float.Parse(x[2]),
                Year = float.Parse(x[3]),
                Month = float.Parse(x[4]),
                Hour = float.Parse(x[5]),
                Holiday = float.Parse(x[6]),
                Weekday = float.Parse(x[7]),
                WorkingDay = float.Parse(x[8]),
                Weather = float.Parse(x[9]),
                Temperature = float.Parse(x[10]),
                NormalizedTemperature = float.Parse(x[11]),
                Humidity = float.Parse(x[12]),
                Windspeed = float.Parse(x[13]),
                Count = float.Parse(x[16])

            })
            .Take(numberOfPredictionsToRead)
            .ToList();
    }
}
