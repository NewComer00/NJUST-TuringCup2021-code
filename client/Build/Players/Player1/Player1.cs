using System.Text;
using Interface;

using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;

using System.Linq;
using System.Text.Json;

namespace SocketClient
{
    class Protocol
    {
        public static readonly byte[] s_header = { 0xaa, 0xaa };
        public static readonly byte[] s_eof = { 0xa5, 0xa5 };
        public static readonly int s_frameLenBytes = 4;

        // frame = s_header + lenHex + payload + s_eof
        public static byte[] WrapPayloadIntoFrame(byte[] payload)
        {
            System.Span<byte> lenHex =
                new System.Span<byte>(new byte[s_frameLenBytes]);
            var frame = new byte[s_header.Length
                + lenHex.Length + payload.Length + s_eof.Length];
            BinaryPrimitives.WriteInt32BigEndian(lenHex, frame.Length);

            s_header.CopyTo(frame, 0);
            lenHex.ToArray().CopyTo(frame, s_header.Length);
            payload.CopyTo(frame, s_header.Length + lenHex.Length);
            s_eof.CopyTo(frame, s_header.Length + lenHex.Length + payload.Length);

            return frame;
        }

        public static byte[] GetPayloadFromFrame(byte[] frame)
        {
            byte[] lenHex = new byte[s_frameLenBytes];
            System.Buffer.BlockCopy(
                frame, s_header.Length, lenHex, 0, s_frameLenBytes);
            int frameLen = BinaryPrimitives.ReadInt32BigEndian(lenHex);
            if (frame.Length == frameLen)
            {
                int payloadLen = frame.Length - s_header.Length
                    - s_frameLenBytes - s_eof.Length;
                byte[] payload = new byte[payloadLen];
                System.Buffer.BlockCopy( frame, s_header.Length + s_frameLenBytes,
                    payload, 0, payloadLen);

                return payload;
            }
            return null;
        }
    }

    class Client
    {
        private readonly Socket sender;

        public Client(string ip_addr, int port)
        {
            // Establish the remote endpoint
            // for the socket. This example
            // uses port 11111 on the local
            // computer.
            IPAddress ipAddr = IPAddress.Parse(ip_addr);
            IPEndPoint localEndPoint = new IPEndPoint(ipAddr, port);

            // Creation TCP/IP Socket using
            // Socket Class Constructor
            sender = new Socket(ipAddr.AddressFamily,
                       SocketType.Stream, ProtocolType.Tcp);

            // Connect Socket to the remote
            // endpoint using method Connect()
            sender.Connect(localEndPoint);
        }

        public int Send(string buffer)
        {
            // Creation of message that
            // we will send to Server
            byte[] messageSent = Encoding.ASCII.GetBytes(buffer);
            byte[] frameSent = Protocol.WrapPayloadIntoFrame(messageSent);
            return sender.Send(frameSent);
        }

        public string Receive(int len)
        {
            // Data buffer
            byte[] messageReceived = new byte[len];

            // We receive the message using
            // the method Receive(). This
            // method returns number of bytes
            // received, that we'll use to
            // convert them to string
            int byteRecv = sender.Receive(messageReceived);
            if(byteRecv > 0)
            {
                int frameBeginIdx = System.MemoryExtensions.IndexOf(
                    System.MemoryExtensions.AsSpan(messageReceived),
                    System.MemoryExtensions.AsSpan(Protocol.s_header));
                int frameEndIdx = System.MemoryExtensions.IndexOf(
                    System.MemoryExtensions.AsSpan(messageReceived),
                    System.MemoryExtensions.AsSpan(Protocol.s_eof));

                if (frameBeginIdx != -1 && frameEndIdx != -1)
                {
                    int frameLen =
                        frameEndIdx - frameBeginIdx + Protocol.s_eof.Length;
                    byte[] frame = new byte[frameLen];
                    System.Buffer.BlockCopy(
                    messageReceived, frameBeginIdx, frame, 0, frameLen);

                    byte[] payload = Protocol.GetPayloadFromFrame(frame);
                    return Encoding.ASCII.GetString(payload, 0, payload.Length);
                }
                return null;
            }
            return null;
        }

        ~Client()
        {
            sender.Shutdown(SocketShutdown.Both);
            sender.Close();
        }

    }
}

namespace RLData
{
    public class ObservationSpace
    {
        public class AgentInfo
        {
            public float CurrentLocationX { get; set; }
            public float CurrentLocationY { get; set; }
        }

        public class Map
        {
            public int Row { get; set; }
            public int Col { get; set; }
            public int[] MapData { get; set; }
        }

        public AgentInfo SelfInfo { get; set; }
        public Map FloorMap { get; set; }
    }

    public class ActionSpace
    {
        public float MoveX { get; set; }
        public float MoveY { get; set; }
        public float FireX { get; set; }
        public float FireY { get; set; }
    }

    public class Reward
    {
        public float Rwrd { get; set; }
    }

    public class GameStatus
    {
        public bool GameFinished { get; set; }
    }
}

namespace PlayerControl
{
    public class Player1 : SuperPlayer
    {
        private class EnvPackage
        {
            public RLData.ObservationSpace ObservationSpace { get; set; }
            public RLData.Reward Reward { get; set; }
            public RLData.GameStatus GameStatus { get; set; }
        }

        private readonly int TOTAL_TIME = 120;
        private readonly double EPS = 1e-6;
        private uint updateCounter;

        private const string LOCALHOST = "127.0.0.1";
        private const int PORT = 11111;
        private SocketClient.Client client;

        private EnvPackage envPackage;

        public override void Awake()
        {
            teamName = "实验室一队";

            updateCounter = 0;
            client = new SocketClient.Client(LOCALHOST, PORT);
        }

        public override void Update()
        {
            try
            {
                var SelfResult = getSelf();
                var FloorsResult = getFloors();
                var CurrentTime = getLeftTime();

                var observationSpace = new RLData.ObservationSpace
                {
                    SelfInfo = new RLData.ObservationSpace.AgentInfo
                    {
                        CurrentLocationX = SelfResult.x,
                        CurrentLocationY = SelfResult.y
                    },
                    FloorMap = new RLData.ObservationSpace.Map
                    {
                        Row = FloorsResult.GetLength(0),
                        Col = FloorsResult.GetLength(1),
                        MapData = FloorsResult.OfType<int>().ToArray()
                    }
                };
                var reward = new RLData.Reward
                {
                    Rwrd = getSelf().totColor - (TOTAL_TIME - CurrentTime)
                    //Rwrd = getSelf().totColor	
                };
                var gameStatus = new RLData.GameStatus
                {
                    GameFinished = (CurrentTime - 116) < EPS
                };
                envPackage = new EnvPackage()
                {
                    ObservationSpace = observationSpace,
                    Reward = reward,
                    GameStatus = gameStatus
                };

                string jsonStringSend = JsonSerializer.Serialize(envPackage);
                client.Send(jsonStringSend);

                string jsonStringRecv = client.Receive(256);
                if(!string.IsNullOrEmpty(jsonStringRecv))
                {
                    var action = 
                        JsonSerializer.Deserialize<RLData.ActionSpace>(jsonStringRecv);
                    moveTo(action.MoveX, action.MoveY);
                    fireTo(action.FireX, action.FireY);
                }

                updateCounter++;
            }
            catch (System.Exception ex)
            {
                string dateTime = System.DateTime.Now.ToString("MM/dd/yyyy HH:mm:ss");
                string info = dateTime + System.Environment.NewLine + ex.ToString();
                System.IO.File.WriteAllText("error_client.log", info);
            }
        }
    }
}
