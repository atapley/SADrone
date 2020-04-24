using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using MLAgents.Sensors;

public class DroneAgent : Agent
{
    public float droneSpeed;
    public float targetSpeed;
    public Transform Target;

    private Rigidbody rBody;
    private Vector3 targetStart;
    private Vector3 agentStart;
    private Vector3 stop;

    System.Random rand = new System.Random();

    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        targetStart = new Vector3(0, .5f, 0);
        agentStart = new Vector3(0, 14, 0);
    }

    public override void OnEpisodeBegin()
    {
        this.transform.localPosition = agentStart;
        Target.localPosition = targetStart;

        int xf = rand.Next(-5, 5);
        int zf = rand.Next(-10, 10);

        stop = new Vector3(xf, .5f, zf);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Do nothing since we are using visual observations
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        //Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = vectorAction[0];
        controlSignal.z = vectorAction[1];
        rBody.AddForce(controlSignal * droneSpeed);

        // Rewards
        float distanceToTarget = Vector3.Distance(this.transform.localPosition,
                                          Target.localPosition);

        // Reached target
        if (distanceToTarget < 13.6f)
        {
            SetReward(1.0f);
        }
    }

    // Update is called once per frame
    void Update()
    {
        // Move our position a step closer to the target.
        float step = targetSpeed * Time.deltaTime; // calculate distance to move

        Target.localPosition = Vector3.MoveTowards(Target.localPosition, stop, step);

        if (Vector3.Distance(Target.localPosition, stop) < 0.001f)
        {
            int xf = rand.Next(-5, 5);
            int zf = rand.Next(-10, 10);

            stop = new Vector3(xf, .5f, zf);
        }
    }

    public override float[] Heuristic()
    {
        var action = new float[2];
        action[0] = Input.GetAxis("Horizontal");
        action[1] = Input.GetAxis("Vertical");
        return action;
    }
}
