#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math, heapq
import rospy, tf
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Header

INF = float('inf')
def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

class PQ(object):
    def __init__(self): self.h=[]
    def push(self,k,s): heapq.heappush(self.h,(k,s))
    def pop(self): return heapq.heappop(self.h)
    def top(self): return self.h[0][0] if self.h else (INF,INF)
    def empty(self): return not self.h

class DStarLite(object):
    def __init__(self, free_fn, w, h):
        self.free = free_fn; self.w=w; self.h=h
        self.km=0; self.rhs={}; self.g={}; self.U=PQ()
        self.start=None; self.goal=None; self.s_last=None
    def n8(self,s):
        x,y=s
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<self.w and 0<=ny<self.h: yield (nx,ny)
    def c(self,a,b):
        if not self.free(b): return INF
        ax,ay=a; bx,by=b
        return 1.4142 if (ax!=bx and ay!=by) else 1.0
    def h(self,s): return manhattan(s,self.start)
    def key(self,s):
        gr=min(self.g.get(s,INF), self.rhs.get(s,INF))
        return (gr + self.h(s) + self.km, gr)
    def initialize(self,start,goal):
        self.start=start; self.goal=goal; self.km=0
        self.s_last=start; self.rhs={goal:0.0}; self.g={}; self.U=PQ()
        self.U.push(self.key(goal), goal)
    def upd(self,u):
        if u!=self.goal:
            self.rhs[u]=min(self.g.get(s,INF)+self.c(u,s) for s in self.n8(u))
        if self.g.get(u,INF)!=self.rhs.get(u,INF):
            self.U.push(self.key(u), u)
    def compute(self, cap=50000):
        it=0
        while (not self.U.empty() and
               (self.U.top()<self.key(self.start) or
                self.rhs.get(self.start,INF)!=self.g.get(self.start,INF))):
            it+=1
            if it>cap: break
            k_old,u = self.U.pop()
            if k_old < self.key(u):
                self.U.push(self.key(u), u)
            elif self.g.get(u,INF) > self.rhs.get(u,INF):
                self.g[u]=self.rhs[u]
                for s in self.n8(u): self.upd(s)
            else:
                self.g[u]=INF
                self.upd(u)
                for s in self.n8(u): self.upd(s)
    def extract(self):
        if self.g.get(self.start,INF)==INF: return []
        path=[self.start]; s=self.start; guard=0
        while s!=self.goal and guard<self.w*self.h:
            guard+=1; best=INF; nxt=None
            for n in self.n8(s):
                c=self.c(s,n)+self.g.get(n,INF)
                if c<best: best=c; nxt=n
            if nxt is None or best==INF: break
            path.append(nxt); s=nxt
        return path

class DStarNode(object):
    def __init__(self):
        rospy.init_node("dstar_planner")
        self.map=None; self.info=None; self.free=None
        self.goal=None; self.odom=None
        self.tf = tf.TransformListener()
        self.pub_path = rospy.Publisher("/dstar/path", Path, queue_size=1, latch=True)
        self.pub_wp   = rospy.Publisher("/controller/waypoints", PoseArray, queue_size=1)
        rospy.Subscriber("/map",  OccupancyGrid, self.on_map)
        rospy.Subscriber("/odom", Odometry,       self.on_odom)
        rospy.Subscriber("/task_planner/next_goal", PoseStamped, self.on_goal)
        rospy.loginfo("D* planner ready.")

    def on_map(self,msg):
        self.map=msg; self.info=msg.info
        w,h=self.info.width,self.info.height
        data=msg.data
        self.free=[False]*(w*h)
        for i,v in enumerate(data):
            self.free[i]=(v>=0 and v<50)  # unknown treated as occupied
        if self.goal and self.odom: self.plan()

    def on_odom(self,msg):
        self.odom=msg
        if self.goal and self.map: self.plan()

    def on_goal(self,msg):
        self.goal=msg
        if self.map and self.odom: self.plan()

    def world_to_grid(self,x,y):
        ix=int((x-self.info.origin.position.x)/self.info.resolution)
        iy=int((y-self.info.origin.position.y)/self.info.resolution)
        ix=max(0,min(self.info.width-1,ix))
        iy=max(0,min(self.info.height-1,iy))
        return (ix,iy)

    def grid_to_world(self,gx,gy):
        x=gx*self.info.resolution+self.info.origin.position.x
        y=gy*self.info.resolution+self.info.origin.position.y
        return (x,y)

    def is_free(self,cell):
        x,y=cell; idx=y*self.info.width+x
        return self.free[idx]

    def current_xy(self):
        try:
            self.tf.waitForTransform("map","base_link",rospy.Time(0),rospy.Duration(0.05))
            (t,q)=self.tf.lookupTransform("map","base_link",rospy.Time(0))
            return (t[0],t[1])
        except:
            p=self.odom.pose.pose.position
            return (p.x,p.y)

    def plan(self):
        sx,sy=self.current_xy()
        gx=self.goal.pose.position.x; gy=self.goal.pose.position.y
        s=self.world_to_grid(sx,sy)
        g=self.world_to_grid(gx,gy)
        w,h=self.info.width,self.info.height
        dsl=DStarLite(self.is_free,w,h)
        dsl.initialize(s,g); dsl.compute()
        cells=dsl.extract()
        self.publish(cells)

    def publish(self,cells):
        hdr=Header(stamp=rospy.Time.now(), frame_id="map")
        # Path for RViz
        path=Path(); path.header=hdr
        for cx,cy in cells:
            x,y=self.grid_to_world(cx,cy)
            ps=PoseStamped(); ps.header=hdr
            ps.pose.position.x=x; ps.pose.position.y=y; ps.pose.orientation.w=1.0
            path.poses.append(ps)
        self.pub_path.publish(path)
        # Waypoints for controller (downsample)
        pa=PoseArray(); pa.header=hdr
        step=max(1,int(len(cells)/50))
        seq=cells[::step]+(cells[-1:] if cells else [])
        for cx,cy in seq:
            x,y=self.grid_to_world(cx,cy)
            p=Pose(); p.position.x=x; p.position.y=y; p.orientation.w=1.0
            pa.poses.append(p)
        self.pub_wp.publish(pa)

if __name__=="__main__":
    DStarNode()
    rospy.spin()
