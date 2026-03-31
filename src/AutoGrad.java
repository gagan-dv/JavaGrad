import java.util.*;

public class AutoGrad {

    public double value;
    public double grad;
    public double exponent;
    public AutoGrad left;
    public AutoGrad right;
    public String op;

    public AutoGrad(double value) {
        this.value = value;
        this.grad = 0;
        this.exponent = 0;
        this.left = null;
        this.right = null;
        this.op = "none";
    }

    public AutoGrad add(AutoGrad other) {
        AutoGrad out = new AutoGrad(this.value + other.value);
        out.left = this;
        out.right = other;
        out.op = "+";
        return out;
    }

    public AutoGrad mul(AutoGrad other) {
        AutoGrad out = new AutoGrad(this.value * other.value);
        out.left = this;
        out.right = other;
        out.op = "*";
        return out;
    }

    public AutoGrad div(AutoGrad other) {
        AutoGrad out = new AutoGrad(this.value / other.value);
        out.left = this;
        out.right = other;
        out.op = "/";
        return out;
    }

    public AutoGrad pow(double exp) {
        AutoGrad out = new AutoGrad(Math.pow(this.value, exp));
        out.left = this;
        out.op = "pow";
        out.exponent = exp;
        return out;
    }

    public AutoGrad relu() {
        AutoGrad out = new AutoGrad(Math.max(0, this.value));
        out.left = this;
        out.op = "relu";
        return out;
    }

    public AutoGrad sigmoid() {
        double s = 1.0 / (1.0 + Math.exp(-this.value));
        AutoGrad out = new AutoGrad(s);
        out.left = this;
        out.op = "sigmoid";
        return out;
    }

    public AutoGrad tanh() {
        double t = Math.tanh(this.value);
        AutoGrad out = new AutoGrad(t);
        out.left = this;
        out.op = "tanh";
        return out;
    }

    public AutoGrad exp() {
        AutoGrad out = new AutoGrad(Math.exp(this.value));
        out.left = this;
        out.op = "exp";
        return out;
    }

    public AutoGrad log() {
        AutoGrad out = new AutoGrad(Math.log(this.value));
        out.left = this;
        out.op = "log";
        return out;
    }

    public void backward() {
        List<AutoGrad> nodes = new ArrayList<>();
        Set<AutoGrad> visited = new HashSet<>();
        build(this, nodes, visited);
        this.grad = 1.0;
        Collections.reverse(nodes);

        for (AutoGrad n : nodes) {
            if (n.op.equals("+")) {
                n.left.grad += n.grad;
                n.right.grad += n.grad;
            }
            else if (n.op.equals("*")) {
                n.left.grad += n.right.value * n.grad;
                n.right.grad += n.left.value * n.grad;
            }
            else if (n.op.equals("/")) {
                n.left.grad += (1.0 / n.right.value) * n.grad;
                n.right.grad += (-n.left.value / (n.right.value * n.right.value)) * n.grad;
            }
            else if (n.op.equals("pow")) {
                n.left.grad += n.exponent * Math.pow(n.left.value, n.exponent - 1) * n.grad;
            }
            else if (n.op.equals("relu")) {
                if (n.value > 0) n.left.grad += n.grad;
            }
            else if (n.op.equals("sigmoid")) {
                n.left.grad += (n.value * (1 - n.value)) * n.grad;
            }
            else if (n.op.equals("tanh")) {
                n.left.grad += (1 - n.value * n.value) * n.grad;
            }
            else if (n.op.equals("exp")) {
                n.left.grad += n.value * n.grad;
            }
            else if (n.op.equals("log")) {
                n.left.grad += (1.0 / n.left.value) * n.grad;
            }
        }
    }

    private void build(AutoGrad node, List<AutoGrad> nodes, Set<AutoGrad> visited) {

        if (visited.contains(node)) {
            return;
        }
        visited.add(node);
        if (node.left != null) {
            build(node.left, nodes, visited);
        }
        if (node.right != null) {
            build(node.right, nodes, visited);
        }
        nodes.add(node);
    }

    @Override
    public String toString() {
        return "AutoGrad(" + value + "), Grad=" + grad;
    }
}
