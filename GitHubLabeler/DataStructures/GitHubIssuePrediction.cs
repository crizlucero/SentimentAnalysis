using Microsoft.ML.Runtime.Api;

namespace GitHubLabeler.DataStructures
{
    internal class GitHubIssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area { get; set; }
    }
}
