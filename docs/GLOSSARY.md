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
One of the numbered chunks of work on the Plan page. The Drive divides the hundred yards evenly among them: F0 and M0 to M9.

### Play record
The one JSON document per play that holds everything the system knows about it: situation, official result, formation before and at the snap, motions, routes, coverage, events, the tackle, and every player's position and role, each with a source and a confidence. It is the product. Clients receive it through the API.

### Motion and shift
Both happen before the snap. A shift is a player moving to a new spot and setting. A motion is a player still moving when the ball is snapped, or moving right up to it: jet (fast across the formation), orbit (looping behind the quarterback), across, short, or return.

### Route tree
The standard numbering of pass routes, 0 to 9: flat, slant, comeback, curl, out, dig, corner, post, go, and a few outside the tree such as wheel, screen and stick. Each receiver's route on a play gets a name, the frame where it broke, and the depth and direction of the break.

### Direction and orientation
Two different angles for every player on every frame. Direction is where they are moving. Orientation is where their body faces. A cornerback backpedalling has a direction pointing downfield and an orientation pointing at the quarterback.

### Coverage family
The defensive shell after the snap, named the way coaches name it: Cover 0 through Cover 6, with whether it was man or zone and which way the safeties rotated.

### Separation
How many yards of space a receiver has from the nearest defender, measured at the moment the ball leaves the quarterback's hand and again at the catch. Read from the defender's side it is closeness: how tightly he stayed on his man.

### Pursuit
For every defender, the distance to whoever has the ball, frame by frame: how close he got, when, and how fast he was closing.

### Pressure
How close the nearest pass rusher got to the quarterback during the dropback, and how long after the snap the first one arrived.

### Correction
A human fix to something the system got wrong: a formation label, a jersey number, a player's position on a frame, the snap frame. Stored beside the machine's answer, never overwritten by a re-run, and used to measure accuracy and to retrain.

### Reviewed
A play a person has looked at in the film room. Reviewed plays show a check; corrected plays a pencil; flagged plays a warning. The counter of reviewed plays per game is the honest measure of how much of the output has been checked.
