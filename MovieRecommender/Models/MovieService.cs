using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MovieRecommender.Models
{
    public partial class MovieService
    {
        public readonly static int _moviesToRecomend = 6;
        public readonly static int _trendingMoviesCount = 20;
        public Lazy<List<Movie>> _movies = new Lazy<List<Movie>>(() => LoadMovieData());
        public List<Movie> _trendingMovies = LoadTrendingMovies();
        public readonly static string _modelPath = @"C:\Users\Christian.Lucero\source\repos\SentimentAnalysis\MovieRecommender\Content\model.zip";

        public static List<Movie> LoadTrendingMovies() =>
            new List<Movie>
            {
                new Movie { MovieID = 1573, MovieName = "Face/Off (1997)" },
                new Movie { MovieID = 1721, MovieName = "Titanic (1997)" },
                new Movie { MovieID = 1703, MovieName = "Home Alone 3 (1997)" },
                new Movie { MovieID = 49272, MovieName = "Casino Royale (2006)" },
                new Movie { MovieID = 5816, MovieName = "Harry Potter and the Chamber of Secrets (2002)" },
                new Movie { MovieID = 3578, MovieName = "Gladiator (2000)" }
            };

        public string GetModelPath() => _modelPath;

        public IEnumerable<Movie> GetSomeSuggestions()
        {
            var movies = GetRecentMovies().ToArray();
            Random rnd = new Random();
            int[] movieSelector = new int[_moviesToRecomend];
            for (int i = 0; i < _moviesToRecomend; i++)
                movieSelector[i] = rnd.Next(movies.Length);

            return movieSelector.Select(s => movies[s]);
        }

        private IEnumerable<Movie> GetRecentMovies() =>
            GetAllMovies().Where(m => m.MovieName.Contains("20")
                || m.MovieName.Contains("198")
                || m.MovieName.Contains("199"));

        public Movie Get(int id) =>
            _movies.Value.Single(m => m.MovieID == id);

        private IEnumerable<Movie> GetAllMovies() =>
            _movies.Value;

        private static List<Movie> LoadMovieData()
        {
            var result = new List<Movie>();
            Stream fileReader = File.OpenRead(@"C:\Users\Christian.Lucero\source\repos\SentimentAnalysis\MovieRecommender\Content\movies.csv");

            StreamReader reader = new StreamReader(fileReader);
            try
            {
                bool header = true;
                while (!reader.EndOfStream)
                {
                    if (header)
                    {
                        reader.ReadLine();
                        header = false;
                    }
                    string[] fields = reader.ReadLine().Split(',');
                    int MovieID = int.Parse(fields[0].ToString().TrimStart(new char[] { '0' }));
                    string MovieName = fields[1];
                    result.Add(new Movie { MovieID = MovieID, MovieName = MovieName });
                }
            }
            finally
            {
                reader?.Dispose();
            }
            return result;
        }
    }
}
