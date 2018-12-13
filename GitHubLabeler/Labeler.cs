using GitHubLabeler.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Octokit;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace GitHubLabeler
{
    class Labeler
    {
        readonly GitHubClient _client;
        readonly string _repoOwner;
        readonly string _repoName;
        readonly string _modelPath;
        readonly MLContext _mlContext;

        readonly PredictionFunction<GitHubIssue, GitHubIssuePrediction> _predFunction;
        readonly ITransformer _trainedModel;
        public Labeler(string modelPath, string repoOwner = "", string repoName = "", string accessToken = "")
        {
            _modelPath = modelPath;
            _repoName = repoName;
            _repoOwner = repoOwner;

            _mlContext = new MLContext(seed: 1);

            using (var stream = new FileStream(_modelPath, System.IO.FileMode.Open, FileAccess.Read, FileShare.Read))
                _trainedModel = _mlContext.Model.Load(stream);

            _predFunction = _trainedModel.MakePredictionFunction<GitHubIssue, GitHubIssuePrediction>(_mlContext);

            if (!string.IsNullOrEmpty(accessToken))
            {
                var productionInformation = new ProductHeaderValue("MLGitHubLabeler");
                _client = new GitHubClient(productionInformation)
                {
                    Credentials = new Credentials(accessToken)
                };
            }
        }

        public void TestPredictionForSingleIssue()
        {
            GitHubIssue singleIssue = new GitHubIssue
            {
                ID = "Any-ID",
                Title = "Entity Framework Crashes",
                Description = "When connectiong to the database, EF is crashing"
            };

            var prediction = _predFunction.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

        public async Task LabelAllNewIssuesInGitHubRepo()
        {
            var newIssues = await GetNewIssues();
            foreach (var issue in newIssues.Where(issue => !issue.Labels.Any()))
            {
                var label = PredictLabel(issue);
                ApplyLabel(issue, label);
            }
        }

        private async Task<IReadOnlyList<Issue>> GetNewIssues()
        {
            var issueRequest = new RepositoryIssueRequest
            {
                State = ItemStateFilter.Open,
                Filter = IssueFilter.All,
                Since = DateTime.Now.AddMinutes(-10)
            };

            var allIssues = await _client.Issue.GetAllForRepository(_repoOwner, _repoName, issueRequest);

            return allIssues.Where(i => !i.HtmlUrl.Contains("/pull/")).ToList();
        }

        string PredictLabel(Issue issue)
        {
            var corefxIssue = new GitHubIssue
            {
                ID = issue.Number.ToString(),
                Title = issue.Title,
                Description = issue.Body
            };
            return Predict(corefxIssue);
        }

        public string Predict(GitHubIssue issue) =>
            _predFunction.Predict(issue).Area;

        void ApplyLabel(Issue issue, string label)
        {
            var issueUpdate = new IssueUpdate();
            issueUpdate.AddLabel(label);
            _client.Issue.Update(_repoOwner, _repoName, issue.Number, issueUpdate);

            Console.WriteLine($"Issue {issue.Number} : \"{issue.Title}\" \t was labeled as {label}");
        }
    }
}
