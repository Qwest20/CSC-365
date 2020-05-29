import java.io.*;
import java.lang.Math;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Scanner;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Main {

    // Method that gives the user a prompt and returns their directory of choice
    public static String userPrompt() {

        // Input scanner so we can work with the keyboard
        Scanner input = new Scanner(System.in);

        // Prompt
        System.out.println("Hi! Would you like to train(0), test(1), or sample(2)?");
        int choice = input.nextInt();
        String n;
        if (choice == 1)
            n = "test";
        else if (choice == 2)
            n = "sample";
        else
            n = "train";
        return n;
    }

    // Method that currently contains all of our new contributions to this project
    public static void sentimentClassification(ExtractWordFeature ewf, String dir) throws IOException {

        // Data Structures for the project

        // A TF HashMap containing the review file names as keys and HashMaps of terms and TF's as values
        // <Review, <Term, TF>>
        HashMap<String, HashMap<String, Double>> TFMap = new HashMap<>();

        // A HashMap to keep record of how many documents a term appears in
        // <Dictionary Term, IDF>
        HashMap<String, Double> docOccurrenceMap = new HashMap<>();

        // A HashMap to keep record of every Dictionary term's IDF
        // <Dictionary Term, IDF>
        HashMap<String, Double> IDFMap = new HashMap<>();

        // A TFIDF HashMap containing the review file names as keys and HashMaps of terms and TDIDF's as values
        // <Review, <Term, TFIDF>>
        HashMap<String, HashMap<String, Double>> TFIDFMap = new HashMap<>();

        // New Dictionary HashMap to be used through out the project
        // <Term, Positivity Score -5 through 5> (scores are saved for use in a future project)
        HashMap<String, Integer> dict = null;
        try {
            dict = ewf.getDictionary();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Main Process!!!

        // Corpus
        File corpus = new File(dir);
        File[] polarities = corpus.listFiles();
        // Records the number of reviews in the project for use further in the project.
        int numReviews=0;
        int numPos=0;
        int numNeg=0;

        // If the corpus isn't empty
        if (polarities != null) {

            // For every polarity File in the corpus Collection (pos, neg)
            for (File polarity : polarities) {

                // Collection of Files in each polarity Collection
                File[] reviews = polarity.listFiles();

                // If the polarity folder isn't empty
                if (reviews != null) {

                    // For every review File in the polarity folder
                    for (File review : reviews) {

                        // TF Crunch Work

                        //Wariables
                        String reviewName = review.getName();
                        numReviews++;
                        if(polarity.getName().equals("pos"))
                            numPos++;
                        else
                            numNeg++;

                        // Arraylists to represent the strings that compose the entire review, and the term words of
                        // the review respectively
                        List<String> reviewWords = ewf.getReviewWords(review);
                        List<String> termWordsOnly = ewf.getTermWords(reviewWords, dict);
                        List<String> distinctTermWordsOnly = termWordsOnly.stream().distinct().collect(Collectors.toList());
                        // HashMap that navigates through the term words only (for speed opt.) to get what it needs
                        HashMap<String, Double> tempTF = ewf.getTFMap(termWordsOnly, reviewWords.size());
                        // Place what we have in the TF HashMap
                        TFMap.put(reviewName, tempTF);

                        // IDF Crunch work Part 1

                        // For loop to iterate through the term words
                        for (String word : distinctTermWordsOnly) {
                            // If first occurrence
                            if (docOccurrenceMap.get(word) == null)
                                // Occurrence = 1
                                docOccurrenceMap.put(word, (double) 1);
                            // If not first
                            else {
                                // Occurrence iterates
                                double currentOccurrenceVal = docOccurrenceMap.get(word);
                                docOccurrenceMap.put(word, ++currentOccurrenceVal);
                            }
                        }
                    }
                }
            }

            // IDF Crunch Work Part 2

            // For loop to iterate through the elements of the Occurrence map
            for (Map.Entry term : docOccurrenceMap.entrySet()) {

                //Wariables from the occurrence HashMap
                String word = term.getKey().toString();
                double docOccurrence = (double) term.getValue();

                // IDF variable calculation
                double IDF = Math.log(numReviews / docOccurrence);

                //Place value into IDF HashMap
                IDFMap.put(word, IDF);
            }

            // TFIDF Crunch Work

            for (File polarity : polarities) {

                // Collection of Files in each polarity Collection
                File[] reviews = polarity.listFiles();

                // If the polarity folder isn't empty
                if (reviews != null) {
                    // For every review File in the polarity folder
                    for (File review : reviews) {

                        // Read the review name
                        String reviewName = review.getName();

                        // Using the file name, retrieve the term HashMap from the greater TFMap
                        HashMap<String, Double> thisReviewTFMap = TFMap.get(reviewName);

                        // Apply the calculation to the TFMap with the IDFMap values for each term
                        // to get the correct TFIDF values.
                        HashMap<String, Double> tempTFIDF = ewf.getTFIDFMap(thisReviewTFMap, IDFMap);

                        // Place what we have in the TF HashMap
                        TFIDFMap.put(reviewName, tempTFIDF);
                    }
                }
            }

            // Logistic Regression Sentiment Analysis (Project 3)-------------------------------------------------------
            
            // Wariables
            double lossSummation = 0;
            // Optimal learningRate is 15 for 98%
            double learningRate = 15;

            // HashMap to keep track of the weights that change through out the program
            // If we're testing, import the weights from the previous run!
            HashMap <String, Double> weights = new HashMap<>();
            if (dir.equals("test")){
                System.out.println("NOTE: The program just started using the weights HashMap!");
                String line;
                BufferedReader reader = new BufferedReader(new FileReader("output/weights.txt"));
                while ((line = reader.readLine()) != null)
                {
                    String[] parts = line.split(":", 2);
                    String key = parts[0];
                    double value = Double.parseDouble(parts[1]);
                    weights.put(key, value);
                }
                reader.close();
            }
            else {
                for (Map.Entry term : dict.entrySet()) {
                    String dictTerm = term.getKey().toString();
                    weights.put(dictTerm, 0.5);
                }
            }
            
            for (File polarity : polarities) {

                // pol wariable to represent the original classification from Stanford
                int pol;
                if(polarity.getName().equals("pos"))
                    pol=1;
                else
                    pol=0;

                // Collection of Files in each polarity Collection
                File[] reviews = polarity.listFiles();

                if (reviews != null) {

                    for (File review : reviews) {

                        // Wariables relative to each review
                        String reviewName = review.getName();
                        double zValue = 0; // z = Summation of weights * TFIDF * dictVal

                        // Derive the tempTFIDF map for each document
                        HashMap<String, Double> thisTFIDFMap = TFIDFMap.get(reviewName);

                        // For calculate the z value in each review
                        for (Map.Entry term : thisTFIDFMap.entrySet()) {

                            // Parse the term to a String
                            String dictWord = term.getKey().toString();

                            // Store the TFIDF value
                            double thisTFIDFVal = (double) term.getValue();

                            // Call back to the current weight for this word from the weights HashMap
                            double weight = weights.get(dictWord);

                            // Increment z value by this numeric with the dict valance in mind
                            int dictionaryScore = dict.get(dictWord);
                            zValue += weight * thisTFIDFVal * dictionaryScore;
                        }

                        // Calculate the positive review probability based on the z value in log regression
                        double posProb = 1 / (1 + Math.exp(-zValue));
                        double pred = Math.round(posProb);
                        double loss;

                        // Originally negative
                        if(pol==0){
                            if(pred==0)
                                loss = 0;
                            else
                                loss = 1;
                        }
                        // Originally positive
                        else{
                            if(pred==1)
                                loss=0;
                            else
                                loss=1;
                        }

                        // Increment the loss summation
                        lossSummation += loss;

                        // Update the weights in the HashMap
                        for (Map.Entry weight : weights.entrySet()) {

                            // Wariables
                            String dictWord = weight.getKey().toString();
                            double oldWeight = (double) weight.getValue();
                            int valance = dict.get(dictWord);
                            double tfidf = 0.0;

                            // Change weight ONLY if this weight word is relevant to this review
                            if (thisTFIDFMap.containsKey(dictWord)) {
                                tfidf = thisTFIDFMap.get(dictWord);
                                // Formula to calculate newWeight
                                double newWeight = oldWeight - (learningRate * tfidf * valance * (pred - pol));
                                weight.setValue(newWeight);
                            }
                        }

                        // Place the review into the new folder!
                        String newPolarity = "";
                        if(pred==1){
                            newPolarity="newPos";
                        }
                        else{
                            newPolarity="newNeg";
                        }
                        Path sourceDirectory = Paths.get("/home/qwest/Programming Stuff/School/Idea Projects/Java/" +
                                "SentimentAnalysisMachineVersion3/"+corpus.getName()+"/"+polarity.getName()+"/"+reviewName);
                        Path targetDirectory = Paths.get("/home/qwest/Programming Stuff/School/Idea Projects/Java/" +
                                "SentimentAnalysisMachineVersion3/target/"+newPolarity+"/"+reviewName);
                        //copy source to target using Files Class
                        Files.copy(sourceDirectory, targetDirectory);
                    }
                }
            }

            // Put the weights HashMap in a textFile
            String outputPath = "output/weights.txt";
            File output = new File(outputPath);
            printToTextFile(output, weights);

            // Print output
            System.out.println("Number Positive: pp+pn = "+ numPos);
            System.out.println("Number Negative: nn+np = "+ numNeg);
            System.out.println("Total Number of Reviews: pp+pn+np+nn = "+numReviews);
            System.out.println("Number of incorrect: pn+np = "+lossSummation);
            System.out.println("Total Loss Percentage: (pn+np)/(pp+pn+np+nn) = "+lossSummation/numReviews);
            System.out.println("Total Accuracy Percentage (inverse of loss) = "+(1-(lossSummation/numReviews)));
        }
    }

    public static void printToTextFile(File f, HashMap<String, Double> output) {
        BufferedWriter bf = null;

        try {

            // create new BufferedWriter for the output file
            bf = new BufferedWriter(new FileWriter(f));

            // iterate map entries
            for (Map.Entry<String, Double> entry : output.entrySet()) {

                // put key and value separated by a colon
                bf.write(entry.getKey() + ":" + entry.getValue());

                // new line
                bf.newLine();
            }

            // send the buffered writer's characters to the appropriate location in the text file
            bf.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }

        finally {

            try {
                // always close the writer, but first, reaffirm the bf's existence
                assert bf != null;
                bf.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    // Main method
    public static void main(String[] args) {

        // Prompt
        String n = userPrompt();

        // New ExtractWordFeature object
        ExtractWordFeature ext = new ExtractWordFeature();

        // Call to sentimentClassification
        try {
            sentimentClassification(ext, n);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
