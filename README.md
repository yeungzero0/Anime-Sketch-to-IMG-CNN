Please edit the config.py, if you are the first play at this program, 
Ensure LOAD_MODEL is False, it look like (line 45):     LOAD_MODEL = False
After program created the "Disc.pth.tar" & "Gen.pth.tar", ensure LOAD_MODEL is Ture.

For creating your own dataset, save the image at /dataset/newIMG/original
Then run "part1_imageResizeTo512png.py" & "part2_imageToSketch.py"
*dataset should at less 500 more IMG, if you want it be more specific way, you should just create it specifically)
*Just like you want to recolor who anime character you want, you should just download which character as the original dataset.
*Then the sketch will recolored to your ref.

dataset - Ensure dataset should 60%/20%/20% (train/val/test)
real:copy with the resized original character image (part1_imageResizeTo512png.py, copy from /dataset/newIMG/to512size)
sketch:copy with the sketch image (part2_imageToSketch.py, copy from /dataset/newIMG/toSketch)
ref:N/A

When you want to play the Sketch to IMG, run "main_runThisForTesting(SketchToImage).py"
