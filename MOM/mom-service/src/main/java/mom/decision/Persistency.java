package mom.decision;

public record Persistency(int warnCount,int highCount) {
    public static Persistency empty(){
        return new Persistency(0,0);
    }
    public Persistency resetWarn(){return new Persistency(0, highCount);}
    public Persistency resetHigh(){return new Persistency(warnCount, 0);}
}
