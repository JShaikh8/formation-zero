# Glossary

Plain-English definitions of the words this project uses. One `### Term` heading per entry,
then one or two sentences. The tracker site renders this file and lets people search it.

### All-22
The NFL's coaches film. Two high, wide camera angles that show all 22 players on every play. Broadcast TV shows you a quarterback and a few receivers; All-22 shows the whole chessboard.

### Sideline angle
The camera on the side of the field, looking across it. Good at showing how far up or down the field players are.

### End zone angle
The camera behind one end zone, looking down the field. Good at showing how far left or right players are. Our film shows each play from both angles, one after the other.

### Play-by-play (PBP)
The official written record of a game: every play, the down, the distance, the yard line, and what happened. We use it as the spine that everything else hangs on.

### Line of scrimmage (LOS)
The imaginary line across the field where the ball is placed before a play. Knowing which yard line it is on is what lets us pin the camera image to the real field.

### Formation
How the offense lines up before the snap. The project names formations by which slots are filled (a receiver wide left, a tight end attached right, and so on) rather than by memorising 96 pictures.

### Personnel grouping
A two-digit code for who is on the field for the offense: running backs then tight ends. "11 personnel" is one running back, one tight end, and therefore three receivers.

### 3-technique
A defensive lineman lined up on the outside shoulder of a guard. The numbering system (0 to 9) describes where each defensive lineman stands relative to the offensive line. A favourite question for the chat: "show me every 3-tech."

### Pre-snap, live, post-whistle
The three phases of a play on film. Formation is read from the last still moment before the snap; movement is read during live action; the pile after the whistle is where the hard questions live.

### Snap detection
Finding the exact frame where the play starts. The offensive line is the best clock in football: it does not move until the ball does.

### Detection
A model drawing a box around each player, referee, and the ball in one frame.

### Tracking
Following each box from frame to frame so the same player keeps the same id. We use ByteTrack for this.

### Field registration
Working out the geometric transform (a homography) between the camera image and the real field, so a pixel becomes a position in yards. The hardest piece of the project.

### Homography
The maths that maps a flat surface seen from an angle (the field in the camera) onto a top-down drawing of it. Eight numbers, solved from a few matched landmarks.

### Bird's-eye view
The top-down 2D drawing of a play, every player as a dot in real yards. It is not a separate feature; it is the tracking table drawn.

### Virtual camera
A 3D replay you can rotate freely, built from player positions and body pose. Like a video game replay, not a re-render of the video.

### Teacher and student models
The teacher is a big, slow, accurate model that labels sample frames. The student is a small, fast model trained on those labels that runs on every frame. Compute is spent so people do not have to draw boxes.

### SAM 3
Meta's Segment Anything Model, third generation. You can prompt it with a word like "football player" and it finds every one in an image or video. Our teacher.

### Hosted vision model
A large model run by a company (Claude, in our case) that we send a few frames to when the local models cannot tell what happened. It costs money per request, which is why there is a budget.

### Router and triggers
The part of the system that decides when to ask the hosted model. Triggers are the rules: a pile of six players, a catch with players from both teams on the ball, a lost ball, an unreadable jersey.

### Proxy rendition
A smaller, cheaper copy of the video (540p) with the same frame numbers as the original. Fast passes run on the proxy; the result addresses the same frame in the full-resolution file.

### Ground truth
Labels a person made by hand. Small, expensive, versioned. Every accuracy number is measured against them.

### Viterbi
An algorithm that finds the most likely sequence of hidden labels given weak evidence per item and strong rules about neighbours. We use it to label camera angles, because adjacent takes alternate.

### nflverse
A free, community-maintained collection of NFL data, including play-by-play with formation and personnel labels for past seasons.

### GSIS
The NFL's Game Statistics and Information System. The NFL API's play records use its ids and stat codes.

### Parquet and DuckDB
Parquet is a file format for tables that reads very fast. DuckDB is a database engine that queries Parquet files directly, with no server to run.

### EPA
Expected points added. A measure of how much a play changed a team's expected score. It lives in nflverse data and the chat can already ask about it.

### Big Data Bowl
Kaggle's annual competition using the NFL's real player-tracking data. That data has the same shape our vision layer will produce, so it lets us test the football logic before the vision is ready.

### Milestone
One of the numbered chunks of work on the Plan page. Ten yards each on the Drive: F0 and M0 to M8 make a hundred-yard field.
