using System;
using System.IO;
using System.IO.Compression;
using System.Net;

namespace SpamDetection.Utils
{
    static class DocumentsUtil
    {
        static string DataDirectoryPath => Path.Combine(Environment.CurrentDirectory, "Data", "spamfolder");
        public static void DownloadFile()
        {
            using (var client = new WebClient())
                client.DownloadFile("https://archive.ics.usi.edu/ml/machine-learning-database/00228/smsspamcollection.zip", "spam.zip");
            ZipFile.ExtractToDirectory("spam.zip", DataDirectoryPath);
        }
    }
}
