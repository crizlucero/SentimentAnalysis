using Microsoft.ML.Runtime.Api;

namespace BikeSharingDemand.DataStructures
{
    public class DemandPrediction
    {
        [ColumnName("Score")]
        public float PredictedCount;
    }
}
