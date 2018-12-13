using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using System;
using System.Collections.Generic;
using System.IO;


namespace MovieRecommender.Models
{
    public class ProfileService
    {
        public List<Profile> _profiles = new List<Profile>(LoadProfileData());
        public int _activeProfileId = -1;

        public List<Tuple<int, int>> GetProfileWatchedMovies(int id)
        {
            foreach (var Profile in _profiles)
                if (id == Profile.ProfileID)
                    return Profile.ProfileMovieRatings;
            return null;
        }

        public Profile GetProfileByID(int id)
        {
            foreach (var Profile in _profiles)
                if (id == Profile.ProfileID)
                    return Profile;
            return null;
        }
        private static IEnumerable<Profile> LoadProfileData()
        {
            var result = new List<Profile>();
            Stream fileReader = File.OpenRead(@"C:\Users\Christian.Lucero\source\repos\SentimentAnalysis\MovieRecommender\Content\profiles.csv");
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
                    int ProfileID = int.Parse(fields[0].ToString().TrimStart(new char[] { '0' }));
                    string ProfileImageName = fields[1];
                    string ProfileName = fields[2];
                    List<Tuple<int, int>> ratings = new List<Tuple<int, int>>();
                    for (int i = 3; i < fields.Length; i += 2)
                        ratings.Add(Tuple.Create(int.Parse(fields[i]), int.Parse(fields[i + 1])));
                    result.Add(new Profile
                    {
                        ProfileID = ProfileID,
                        ProfileImageName = ProfileImageName,
                        ProfileName = ProfileName,
                        ProfileMovieRatings = ratings
                    });
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
