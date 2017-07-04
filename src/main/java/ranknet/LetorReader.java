package ranknet;

import com.sun.tools.javac.util.Pair;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class LetorReader {
    private final Scanner scanner;
    private final int featureCount;

    private Data currentData;
    private long currentQID;

    public LetorReader(File file, int featureCount) throws FileNotFoundException {
        this.scanner = new Scanner(file);
        this.featureCount = featureCount;
    }

    public boolean hasNext() {
        return scanner.hasNextLine();
    }

    public List<Data> next() {
        List<Data> dataSet = new LinkedList<Data>();

        if (currentData == null) {
            Pair<Long, Data> queryData = getNextData();
            currentQID = queryData.fst;
            currentData = queryData.snd;
        }

        dataSet.add(currentData);

        boolean isSameQID = true;
        while (isSameQID && scanner.hasNextLine()) {
            Pair<Long, Data> queryData = getNextData();
            long qID = queryData.fst;
            Data data = queryData.snd;

            if (qID == currentQID) {
                dataSet.add(data);
            } else {
                currentData = data;
                currentQID = qID;
                isSameQID = false;
            }
        }

        return dataSet;
    }

    private Pair<Long, Data> getNextData() {
        String line = scanner.nextLine();
        String[] tokens = line.split("\\s");

        double rankScore = Double.valueOf(tokens[0]);
        long qID = Long.valueOf(tokens[1].split(":")[1]);

        double[] features = new double[featureCount];
        for (int i = 0; i < featureCount; i++) {
            features[i] = Double.valueOf(tokens[2 + i].split(":")[1]);
        }

        Data data = new Data(rankScore, Nd4j.create(features, new int[] { 1, featureCount }), 0);

        return new Pair<Long, Data>(qID, data);
    }
}
