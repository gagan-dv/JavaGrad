public class Demo {
    public static void main(String[] args) {

        AutoGrad a = new AutoGrad(3.0);
        AutoGrad b = new AutoGrad(2.0);
        AutoGrad c = new AutoGrad(4.0);
        AutoGrad d = a.mul(b).add(c);
        System.out.printf("d  = %.2f%n", d.value);
        d.backward();
        System.out.printf("da = %.4f%n", a.grad);
        System.out.printf("db = %.4f%n", b.grad);
        System.out.printf("dc = %.4f%n", c.grad);

    }
}
