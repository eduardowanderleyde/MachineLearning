import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
import weka.experiment.Experiment;
import weka.experiment.InstanceQuery;

public class WEKAExperiment {
    public static void main(String[] args) throws Exception {
        // Carregando o conjunto de dados
        DataSource source = new DataSource("caminho/do/arquivo.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Definindo os algoritmos
        Classifier[] classifiers = new Classifier[]{
                new ZeroR(),
                new Logistic(),
                new IBk(),
                new J48(),
                new NaiveBayes(),
                new RandomForest()
        };

        // Realizando a validação cruzada
        for (Classifier classifier : classifiers) {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new java.util.Random(1));
            System.out.println("Resultado para o classificador " + classifier.getClass().getSimpleName());
            System.out.println(eval.toSummaryString());
        }
    }
}
