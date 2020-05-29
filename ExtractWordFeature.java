import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Scanner;

public class ExtractWordFeature {

    // getReviewWords method returns the words of the review file as a String array
    public ArrayList<String> getReviewWords(File Review) throws IOException {

        // Input scanner to read the file through
        Scanner input = new Scanner(Review);

        // A string ArrayList that allows us to add as many words as desired, no matter what the length of the blog is
        ArrayList<String> words = new ArrayList<String>();

        // While there are still more words in the blog...
        while (input.hasNext()) {

            // Add the word to the list
            words.add(input.next());
        }

        // For the amount of words in this list...
        for (int i = 0; i < words.size(); i++) {

            // Remove punctuation and such, but keep hyphens and apostrophes
            words.set(i, words.get(i).replaceAll("[^a-zA-Z-]+", ""));

        }

        // return the ArrayList
        return words;
    }

    // getDictionary method returns the dictionary words as a hashmap with scores as values
    public HashMap<String, Integer> getDictionary() throws IOException {

        // Dictionary text file (the one with no spaces)
        File inputFile = new File("NoSpaceDictionary.txt");

        // Input scanner to read the file through
        Scanner input = new Scanner(inputFile);

        // ArrayLists to help us read in words and scores respectively
        ArrayList<String> words = new ArrayList<String>();
        ArrayList<Integer> scores = new ArrayList<Integer>();

        // Our Dictionary HashMap that uses Strings for keys and Integers for values
        HashMap<String, Integer> dictionary = new HashMap<String, Integer>();

        // A count to keep track of our progress in the ArrayList (keeps track of Strings vs ints)
        int count = 0;

        // While there are still elements in the file
        while (input.hasNext()) {

            // We have a word and we add it to words
            if (count % 2 == 0) {
                String w = input.next();
                words.add(w);
            }
            // We have a value and we add it to scores
            else {
                int v = Integer.parseInt(input.next());
                scores.add(v);
            }
            count++;
        }

        // Fill the HashMap with words and scores
        for (int i = 0; i < words.size(); i++) {
            dictionary.put(words.get(i), scores.get(i));
        }

        // Return the HashMap
        return dictionary;
    }

    // New Methods!

    // Get a String ArrayList that includes ONLY the term words
    public ArrayList <String> getTermWords(List<String> reviewWords, HashMap<String, Integer> Dictionary){

        // A String ArrayList for the term words only
        ArrayList <String> termWordsOnly= new ArrayList<>();

        // For loop to iterate through the established reviewWords
        for (String word : reviewWords) {

            // If the word belongs to the dictionary
            if (Dictionary.containsKey(word)) {

                // Add it to this new ArrayList
                termWordsOnly.add(word);
            }
        }
        return termWordsOnly;
    }

    // getTF returns our TF values for each review file (new method for this project!)
    public HashMap<String, Double> getTFMap(List<String> termWords, double reviewSize){

        // A HashMap to collect the frequencies
        HashMap<String, Double> freqMap = new HashMap<>();
        HashMap<String, Double> outputMap = new HashMap<>();

        // Enhanced For loop to go through the list "blog" for each "word"
        for (String word : termWords) {

            // If this is first occurrence of element
            if (freqMap.get(word) == null)
                freqMap.put(word, (double) 1);

            // If the element already exists in the hash map
            else {

                // Increment its frequency
                double freq = freqMap.get(word);
                freqMap.put(word, ++freq);
            }
        }

        // For loop to iterate through the freqMap
        for (Map.Entry term : freqMap.entrySet()) {

            // Wariables
            String word = term.getKey().toString();
            double freq = (double) term.getValue();

            // TF calculation
            double TF = freq/reviewSize;

            // Place this as an element in the HashMap
            outputMap.put(word, TF);
        }
        return outputMap;
    }

    // getTF returns our TFIDF values for each review file (new method for this project!)
    public HashMap<String, Double> getTFIDFMap(HashMap <String, Double> docTFMap, HashMap <String, Double> IDFMap){

        // A HashMap to collect the frequencies
        HashMap<String, Double> outputMap = new HashMap<>();

        // A for loop to iterate through this document's TF Map
        for (Map.Entry term : docTFMap.entrySet()) {

            // Wariables
            String word = term.getKey().toString();
            double TF = (double) term.getValue();
            double IDF = IDFMap.get(word);

            // TFIDF calculation
            double TFIDF = TF*IDF;

            // Insert these as an element in the HashMap
            outputMap.put(word, TFIDF);
        }
        return outputMap;
    }
}