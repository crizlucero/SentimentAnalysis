using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using MovieRecommender.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace MovieRecommender.Controllers
{
    public class StringTable
    {
        public string[] ColumnNames { get; set; }
        public string[,] Values { get; set; }
    }
    public class MoviesController : Controller
    {
        readonly MovieService _movieService;
        readonly ProfileService _profileService;
        readonly AppSettings _appSettings;
        readonly ILogger<MoviesController> _logger;

        public MoviesController(ILogger<MoviesController> logger, IOptions<AppSettings> appSettings)
        {
            _movieService = new MovieService();
            _profileService = new ProfileService();
            _logger = logger;
            _appSettings = appSettings.Value;
        }
        public IActionResult Choose() =>
            View(_movieService.GetSomeSuggestions());

        static async Task<string> InvokeRequestResponseService(int id, ILogger logger, AppSettings appSettings) =>
            null;

        public ActionResult Recommend(int id)
        {
            Profile activeProfile = _profileService.GetProfileByID(id);

            var ctx = new MLContext();

            ITransformer loadedModel;
            using (var stream = new FileStream(_movieService.GetModelPath(), FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = ctx.Model.Load(stream);

            var predictionFunction = loadedModel.MakePredictionFunction<RatingData, RatingPrediction>(ctx);
            List<Tuple<int, float>> ratings = new List<Tuple<int, float>>();
            List<Tuple<int, int>> MovieRatings = _profileService.GetProfileWatchedMovies(id);
            List<Movie> WatchedMovies = new List<Movie>();

            foreach (Tuple<int, int> tuple in MovieRatings)
                WatchedMovies.Add(_movieService.Get(tuple.Item1));

            RatingPrediction prediction = null;
            foreach (var movie in _movieService._trendingMovies)
            {
                prediction = predictionFunction.Predict(new RatingData
                {
                    userId = id.ToString(),
                    movieId = movie.MovieID.ToString()
                });
                float normalizedScore = Sigmoid(prediction.Score);

                ratings.Add(Tuple.Create(movie.MovieID, normalizedScore));
            }
            ViewData["watchedmovies"] = WatchedMovies;
            ViewData["ratings"] = ratings;
            ViewData["trendingmovies"] = _movieService._trendingMovies;
            return View(activeProfile);
        }

        float Sigmoid(float x) => (float)(100 / (1 + Math.Exp(-x)));

        public ActionResult Watch() => View();

        public ActionResult Profiles() => View(_profileService._profiles);

        public ActionResult Watched(int id)
        {
            Profile activeProfile = _profileService.GetProfileByID(id);
            List<Tuple<int, int>> MovieRatings = _profileService.GetProfileWatchedMovies(id);
            List<Movie> WatchedMovies = new List<Movie>();

            foreach (Tuple<int, int> tuple in MovieRatings)
                WatchedMovies.Add(_movieService.Get(tuple.Item1));
            ViewData["watchedmovies"] = WatchedMovies;
            ViewData["trendingmovies"] = _movieService._trendingMovies;
            return View(activeProfile);
        }

        public class JsonContent : StringContent
        {
            public JsonContent(object obj) : base(JsonConvert.SerializeObject(obj), Encoding.UTF8, "application/json") { }
        }

        public class RatingData
        {
            [Column("0")]
            public string userId;
            [Column("1")]
            public string movieId;
            [Column("2")]
            [ColumnName("Label")]
            public float Label;
        }

        public class RatingPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool predictedLabel;
            public float Score;
        }
    }


}