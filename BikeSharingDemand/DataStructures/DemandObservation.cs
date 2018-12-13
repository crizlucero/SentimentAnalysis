namespace BikeSharingDemand.DataStructures
{
    public class DemandObservation
    {
        public float Season { get; set; }
        public float Year { get; set; }
        public float Month { get; set; }
        public float Hour { get; set; }
        public float Holiday { get; set; }
        public float Weekday { get; set; }
        public float WorkingDay { get; set; }
        public float Weather { get; set; }
        public float Temperature { get; set; }
        public float NormalizedTemperature { get; set; }
        public float Humidity { get; set; }
        public float Windspeed { get; set; }
        public float Count { get; set; }
    }

    public static class DemandObservationSample
    {
        public static DemandObservation SingleDemandSampleData =>
            new DemandObservation
            {
                Season = 3,
                Year = 1,
                Month = 8,
                Hour = 10,
                Holiday = 0,
                Weekday = 4,
                WorkingDay = 1,
                Weather = 1,
                Temperature = 0.8f,
                NormalizedTemperature = 0.7576f,
                Humidity=0.55f,
                Windspeed = 0.2239f
            };
    }
}
