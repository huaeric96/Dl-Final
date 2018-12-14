import csv
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

with open("labels.csv", "w") as labels:
	csvwriter = csv.writer(labels, delimiter=",")
	for genre in genres:
		for num in range(0,100):
			if num < 10:
				song = genre + "_0000" + str(num) + ".png"
			else:
				song = genre + "_000" + str(num) + ".png"
			csvwriter.writerow([song, genre])
			print("Song ", song, "genre", genre)