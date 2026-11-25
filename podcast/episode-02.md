# Episode 02: A sharper ruler, and what it measured

This morning I started two matches side by side. One was today's teacher. The other stopped its playouts early and asked the network who was winning. By lunch, both were winning about sixty two percent of their games against plain c2. That confirms the teacher is strong. But it also showed me a problem with how I was measuring.

The difference between the two was less than one point. And after sixty kingdoms, the uncertainty was ten points either way. Even three hundred kingdoms would only get that down to about four. The changes I am testing are worth one to three points. So a whole match, which takes ten hours, could not tell them apart. I stopped the early-stopping match, and the plan for four more matches went with it.

Instead, I built a sharper ruler. I took two hundred and forty real decisions from c2's own games. There are buys, and there are action plays. For each one, I played every legal option to the end of the game two thousand and forty eight times. That is the truth, as close as this laptop can get to it. Then I asked each version of the teacher the same questions, three times each, and measured how much win chance its choice gives up compared with the truth. I call that the regret.

Here is what it found. The c2 network on its own gives up almost nine tenths of a point of win chance per decision. That sounds small, but there are about thirty five real decisions per game. Today's teacher brings it down to just under six tenths.

Stopping early did not hold up. With six hundred and forty playouts it was a little better than today's teacher. But with sixteen hundred it was no better at all. It hit a floor at about half a point. That floor is the network's own misjudgement of positions. More playouts cannot fix a biased judge. So I have dropped that idea.

Full playouts kept improving. With three hundred and twenty draws, which is what three processes give the advisor, the regret fell to under half a point.

And the biggest surprise was the override rule. Today the teacher only overrides when its gain is bigger than two standard errors. That turns out to be too careful. With one standard error, the regret fell at every budget I tried. With three hundred and twenty draws it reached about four tenths of a point. That is less than half of what c2 gives up on its own. I checked that this was not one lucky setting. The whole table is smooth around it. I have already changed the advisor to use the new rule.

One more thing looks promising. The teacher now also checks the card choices inside your actions. That means what to gain with a Workshop or an Artisan, what to trash with a Remodel, and what to play twice with a Throne Room. In the match, that version is ahead of today's teacher by about five points after seventy kingdoms. That is not certain yet. The uncertainty still reaches just below zero.

So, next. One match now plays the full candidate. It uses three hundred and twenty draws, the new override rule, and the card choices. It plays the same kingdoms as today's teacher, so the two can be compared game by game. If it holds up, it goes into the advisor for your next session.
