using ImageClassification.ModelScorer;
using System;
using System.IO;

namespace ImageClassification
{
    public class Program
    {
        static void Main(string[] args)
        {
            string assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");

            string tagsTsv = Path.Combine(assetsPath, "inputs", "images", "tags.tsv");
            string imagesFolder = Path.Combine(assetsPath, "inputs", "images");
            string inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
            string lbls = Path.Combine(assetsPath, "inputs", "inception", "imagenet_comp_graph_label_strings.txt");

            var customInceptionPb = Path.Combine(assetsPath, "inputs", "inception_custom", "model_tf.pb");
            var customLabelsTxt = Path.Combine(assetsPath, "inputs", "inception_custom", "labels.txt");

            try
            {
                var modelScorer = new TFModelScorer(tagsTsv, imagesFolder, inceptionPb, lbls);
                modelScorer.Score();
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error {e.Message}");
            }

            Console.ReadLine();
        }
    }
}
