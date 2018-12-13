using System;
using System.IO;
using System.Linq;

namespace ImageClassification.ModelScorer
{
    public static class ModelHelpers
    {
        static FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);

        public static string GetAssetsPath(params string[] paths)
        {
            if (paths == null || paths.Length == 0)
                return null;
            return Path.Combine(paths.Prepend(_dataRoot.Directory.FullName).ToArray());
        }

        public static string[] ReadLabels(string labelsLocation) =>
            File.ReadAllLines(labelsLocation);

        public static (string, float) GetBestLabel(string[] labels, float[] probs)
        {
            var max = probs.Max();
            var index = probs.AsSpan().IndexOf(max);
            return (labels[index], max);
        }
    }
}