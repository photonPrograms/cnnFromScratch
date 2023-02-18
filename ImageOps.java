import java.util.*;
import java.io.*;
import java.awt.image.*;
import javax.imageio.*;

public class ImageOps {
    /* for various image operations on the png mnist images
     * such as reading and conversion to matrix
     */

     public static BufferedImage openImage(String filePath) throws IOException {
        /* open the image with the given file name
         * params:
         * : filePath: the path relative to source code directory
         */

         BufferedImage image = ImageIO.read(new File(filePath));
         return image;
     }


     public static double[][] convertImageToMatrix(BufferedImage image) {
        /* convert the image to a 2d array
         * params:
         * : image: the BufferedImage to be converted 
         */
        int rows = image.getWidth(), cols = image.getHeight();

        // a vector of pixels
        int[] imageVector = image.getRGB(0, 0, rows, cols, null, 0, rows);
        
        double[][] matrix = new double[rows][cols];
        int i = 0, j = 0;
        for (int pixel: imageVector) {
            // normalized to [-0.5, 0.5]
            matrix[i][j] = (double) (((int) pixel >> 16 & 0xff)) / 255.0 - 0.5;
            j++;
            if (j == cols) {
                j = 0;
                i++;
            }
        }

        return matrix;
     }

     public static double[][] getImageMatrix(String filePath) throws IOException {
        /* encapsulates openImage() and getImageMatrix()
         * params:
         * : filePath: the relative address of the image file
         */
        return convertImageToMatrix(openImage(filePath));
     }
}