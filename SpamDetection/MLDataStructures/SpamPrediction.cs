using Microsoft.ML.Runtime.Api;

namespace SpamDetection.MLDataStructures
{
    class SpamPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsSpam { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }
}
