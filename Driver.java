import java.io.IOException;
import java.awt.image.*;

public class Driver {
    public static void main(String[] args) throws IOException {
        CNN cnn = new CNN(
            new Conv3x3(8),
            new MaxPooling(2),
            new Dense(13 * 13 * 8, 10)
        );
        cnn.compile(5e-4);

        cnn.train(10, 20000);
        cnn.test(5000);
        cnn.saveSeries();

        // cnn.train(15, 40000);
        // cnn.test(8000);
        // cnn.saveSeries();
    }
}
