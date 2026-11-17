// Pixel-based hit targets stay usable when zooming a dense constellation.
export function pickGlobeNode({THREE,camera,positions,nodes,pointer,width,height,radius=9}) {
 const earth=new THREE.Sphere(new THREE.Vector3(),6.35),hit=new THREE.Vector3();
 let nearest=null,best=radius*radius;
 for(const node of nodes){
  const position=positions.get(node.node_id);if(!position)continue;
  const projected=position.clone().project(camera);if(projected.z<-1||projected.z>1)continue;
  const x=(projected.x+1)*width/2,y=(1-projected.y)*height/2,distance=(x-pointer.x)**2+(y-pointer.y)**2;
  if(distance>best)continue;
  const ray=new THREE.Ray(camera.position,position.clone().sub(camera.position).normalize());
  const blocked=ray.intersectSphere(earth,hit);
  if(blocked&&hit.distanceTo(camera.position)<position.distanceTo(camera.position)-.035)continue;
  nearest=node.node_id;best=distance;
 }
 return nearest;
}
