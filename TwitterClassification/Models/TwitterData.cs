using Microsoft.ML.Runtime.Api;

namespace TwitterClassification.Models
{
    class TwitterData
    {
        [Column(ordinal: "0", name: "Label")]
        public float sentiment { get; set; }
        [Column(ordinal: "1")]
        public string content { get; set; }
    }

    public class TwitterPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        [ColumnName("Probability")]
        public float Probability { get; set; }
        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
