Thinking Process:

1.  **Analyze Current State (Model C):**
    *   **Status**: Active, using NewsAPI.
    *   **Data Source**: General news headlines (e.g., "Man City transfer news", "Liverpool injury update").
    *   **Pros**: Free (currently), covers major events (injuries, transfers).
    *   **Cons**: Latency (news articles take time to write), less "raw emotion" than social media.

2.  **Evaluate Instagram/Facebook:**
    *   **Purpose**: Capture fan sentiment, player lifestyle/morale updates.
    *   **Feasibility (Instagram):**
        *   *Official API*: Focused on "Business" accounts (managing your own). Hard to scrape random hashtags/teams without approval.
        *   *Scraping*: Extremely difficult. Meta is aggressive against scrapers.
        *   *Value*: High for visual/morale (e.g., player partying before match), but NLP on images is hard. Comments are noisy.
    *   **Feasibility (Facebook):**
        *   *Official API*: Restricted. Public Page content access is possible but requires App Review.
        *   *Value*: Older demographic, often very noisy/spammy fan groups.
    *   **Feasibility (Reddit):**
        *   *API*: Accessible, structured (r/soccer, r/Gunners). Good middle ground.
    *   **Feasibility (Twitter/X):**
        *   *API*: The "Gold Standard" for sports. Real-time. But expensive ($100/mo for basic tier).

3.  **Recommendations:**
    *   **Don't add Instagram/FB directly via official APIs.** Too much overhead/maintenance for a betting bot.
    *   **Better Alternative**: Reddit. It has a free/easy API, passionate communities, and faster reaction time than NewsAPI.
    *   **Best Paid Alternative**: Twitter/X API (if budget allows).

4.  **Design for "Social Media Expansion" (if requested):**
    *   Architecture: `SentimentFetcher` class is already extensible.
    *   New method: `_fetch_reddit(self, team_name)` using `praw` library.
    *   Fusion: Aggregate scores from News + Reddit.

5.  **Model C Analysis (based on test output):**
    *   *To be filled after seeing test output.*
