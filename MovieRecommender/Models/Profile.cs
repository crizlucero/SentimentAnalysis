using System;
using System.Collections.Generic;

namespace MovieRecommender.Models
{
    public class Profile
    {
        public int ProfileID { get; set; }
        public string ProfileImageName { get; set; }
        public string ProfileName { get; set; }
        public List<Tuple<int, int>> ProfileMovieRatings { get; set; }
    }
}
