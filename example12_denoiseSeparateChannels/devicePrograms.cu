// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "gdt/random/random.h"

using namespace osc;

#define NUM_LIGHT_SAMPLES 1//光源个数
#define NUM_RANDOM_SAMPLES 1

namespace osc {

  typedef gdt::LCG<16> Random;
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  /*! per-ray data now captures random number generator, so programs
      can access RNG state */
  struct PRD {
    Random random;
    vec3f  pixelColor;
    vec3f  pixelNormal;
    vec3f  pixelAlbedo;//漫反射系数
	vec3f feedback;
	bool hit;
	int bounce;
  };//像素的属性
  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------


  extern "C" __constant__ float PI = 3.1415926;
  extern "C" __constant__ float default_F0 = 0.04;
  extern "C" __constant__ float default_alpha = 0.5;
  static __forceinline__ __device__
	 vec3f  half_vector(vec3f view, vec3f light){
	  vec3f half = gdt::normalize(view + light);
	  return half;
  }
  static __forceinline__ __device__
	  vec3f specialmatch(vec3f a, vec3f b) {
	  vec3f output;
	  output.x = a.x * b.x;
	  output.y = a.y * b.y;
	  output.z = a.z * b.z;
	  return output;
  }
  static __forceinline__ __device__
	float NDF(vec3f normal, vec3f half, float alpha){
	  float a2 = alpha * alpha;
	  float NdotH = gdt::dot(normal, half);
	  if (NdotH < 0.0) NdotH = 0.0;
	  float NdotH2 = NdotH * NdotH;
	  float nom = a2;
	  float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	  denom = PI * denom * denom;
	  return nom / denom;
  }
  static __forceinline__ __device__
	float GeometrySchlickGGX(float NdotV, float k){
	  float nom = NdotV;
	  float denom = NdotV * (1.0 - k) + k;
	  return nom / denom;
  }
  static __forceinline__ __device__
	  float K_IBL(float alpha) {
	  float k = alpha * alpha / 2;
	  return k;
  }
  static __forceinline__ __device__
	float GeometrySmith(vec3f normal, vec3f view, vec3f light, float alpha){
	  float k = K_IBL(alpha);
	  float NdotV = gdt::dot(normal, view);
	  if(NdotV < 0.0) NdotV = 0.0;
	  float NdotL = gdt::dot(normal, light);
	  if(NdotL < 0.0) NdotL = 0.0;
	  float ggx1 = GeometrySchlickGGX(NdotV, k);
	  float ggx2 = GeometrySchlickGGX(NdotL, k);
	  return ggx1 * ggx2;
  }
  static __forceinline__ __device__
	vec3f fresnelSchlick(vec3f half, vec3f view, vec3f F0){
	  float HdotV = gdt::dot(half, view);
	  return F0 + (vec3f(1.0) - F0) * (float)pow(1.0 - HdotV, 5.0);
  }
  static __forceinline__ __device__
	 vec3f BRDF(vec3f view, vec3f light, vec3f normal, vec3f F0, float alpha, vec3f color, vec3f kd, vec3f ks) {
	  vec3f v = gdt::normalize(view);
	  vec3f l = gdt::normalize(light);
	  vec3f n = gdt::normalize(normal);
	  vec3f h = half_vector(v, l);
	  vec3f former = specialmatch(color, kd) / PI;
	  float D = NDF(n, h, alpha);
	  float G = GeometrySmith(n, v, l, alpha);
	  vec3f F = fresnelSchlick(h, v, F0);
	  float VdotN = gdt::dot(v, n);
	  float LdotN = gdt::dot(l, n);
	  vec3f latter = specialmatch(D * G * F, ks) / (4 * VdotN * LdotN);
	  return former + latter;
  }


  
  extern "C" __global__ void __closesthit__shadow()
  {
    /* not going to be used ... */
  }
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    PRD &former_prd = *getPRD<PRD>();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;


	vec3f Kd;
	if(sbtData.Kd == vec3f(0.0)) Kd = vec3f(1.0);
	else Kd = sbtData.Kd;
	const vec3f Ks = sbtData.Ks;


    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];
    vec3f Ng = cross(B-A,C-A);
    vec3f Ns = (sbtData.normal)
      ? ((1.f-u-v) * sbtData.normal[index.x]
         +       u * sbtData.normal[index.y]
         +       v * sbtData.normal[index.z])
      : Ng;
    
    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const vec3f rayDir = optixGetWorldRayDirection();
    
    if (dot(rayDir,Ng) > 0.f) Ng = -Ng;
    Ng = normalize(Ng);
    
    if (dot(Ng,Ns) < 0.f)
      Ns -= 2.f*dot(Ng,Ns)*Ng;
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available（计算物体本身的反射光颜色）
    // ------------------------------------------------------------------
    vec3f diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord) {
      const vec2f tc
        = (1.f-u-v) * sbtData.texcoord[index.x]
        +         u * sbtData.texcoord[index.y]
        +         v * sbtData.texcoord[index.z];
      
      vec4f fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
      diffuseColor *= (vec3f)fromTexture;
    }

    // start with some ambient term
	vec3f dirlight = 0.f;//(0.1f + 0.2f*fabsf(dot(Ns,rayDir)))*diffuseColor;//像素颜色
    
    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const vec3f surfPos
      = (1.f-u-v) * sbtData.vertex[index.x]
      +         u * sbtData.vertex[index.y]
      +         v * sbtData.vertex[index.z];
	const vec3f view = gdt::normalize(optixLaunchParams.camera.position - surfPos);
    const int numLightSamples = NUM_LIGHT_SAMPLES;
    for (int lightSampleID=0;lightSampleID<numLightSamples;lightSampleID++) {
      // produce random light sample（随机生成采样）
      const vec3f lightPos
        = optixLaunchParams.light.origin
        + former_prd.random() * optixLaunchParams.light.du
        + former_prd.random() * optixLaunchParams.light.dv;
      vec3f lightDir = lightPos - surfPos;
      float lightDist = gdt::length(lightDir);
      lightDir = normalize(lightDir);
	  const float area = gdt::length(gdt::cross(optixLaunchParams.light.du, optixLaunchParams.light.dv));
	  const vec3f Nl = gdt::normalize(gdt::cross(optixLaunchParams.light.du, optixLaunchParams.light.dv));
	  const float NldotL = fabs(gdt::dot(Nl, lightDir));
	  vec3f fr = 1.0f;// BRDF(view, lightDir, Ns, vec3f(default_F0), default_alpha, diffuseColor, Kd, Ks);
      // trace shadow ray:
      const float NdotL = dot(lightDir,Ns);
      if (NdotL >= 0.f) {
        vec3f lightVisibility = 0.f;
        // the values we store the PRD pointer in:
        uint32_t u0, u1;
        packPointer( &lightVisibility, u0, u1 );
        optixTrace(optixLaunchParams.traversable,
                   surfPos + 1e-3f * Ng,
                   lightDir,
                   1e-3f,      // tmin
                   lightDist * (1.f-1e-3f),  // tmax
                   0.0f,       // rayTime
                   OptixVisibilityMask( 255 ),
                   // For shadow rays: skip any/closest hit shaders and terminate on first
                   // intersection with anything. The miss shader is used to mark if the
                   // light was visible.
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT
                   | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                   | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                   SHADOW_RAY_TYPE,            // SBT offset
                   RAY_TYPE_COUNT,               // SBT stride
				   SHADOW_RAY_TYPE,            // missSBTIndex 
                   u0, u1 );
        dirlight
          += lightVisibility
          *  optixLaunchParams.light.power
          *  specialmatch(diffuseColor, fr)
          *  (NdotL * NldotL * area / (lightDist*lightDist*numLightSamples));
      }
    }
	const int numRandomSample = NUM_RANDOM_SAMPLES;
	vec3f indirlight = 0.f;
	PRD my_prd;
	my_prd.bounce = former_prd.bounce + 1;
	my_prd.random = former_prd.random;
	uint32_t u0, u1;
	packPointer(&my_prd, u0, u1);
	const float distance = gdt::length(optixLaunchParams.light.origin - surfPos);
	for (int randomSampleID = 0; randomSampleID < numRandomSample; randomSampleID++) {
		vec3f random_reflect_dir = 0.f;
		while (gdt::dot(random_reflect_dir, Ns) <= 0.0f) {
			random_reflect_dir.x = my_prd.random();
			random_reflect_dir.y = my_prd.random();
			random_reflect_dir.z = my_prd.random();
		}
		vec3f lightDir = normalize(random_reflect_dir);
		optixTrace(optixLaunchParams.traversable,
			surfPos + 1e-3f * Ng,
			lightDir,
			1e-3f,      // tmin
			distance * (1.f - 1e-3f),
			0.0f,       // rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
			RADIANCE_RAY_TYPE,            // SBT offset
			RAY_TYPE_COUNT,               // SBT stride
			RADIANCE_RAY_TYPE,            // missSBTIndex 
			u0, u1);
		if (my_prd.hit) {
			const float NdotL = gdt::dot(Ns, lightDir);
			vec3f fr = 1.0f;//BRDF(view, lightDir, Ns, vec3f(default_F0), default_alpha, diffuseColor, Kd, Ks);
			indirlight += specialmatch(fr, my_prd.feedback) * NdotL / numRandomSample;
		}
	}
	vec3f pixelColor = dirlight + indirlight;
	if (former_prd.bounce == 0) {
		former_prd.pixelNormal = Ns;
		former_prd.pixelAlbedo = diffuseColor;
		former_prd.pixelColor = pixelColor;
	}
	else {
		former_prd.hit = true;
		former_prd.feedback = pixelColor;
	}
  }
  
  extern "C" __global__ void __anyhit__radiance(){
	  PRD &prd = *getPRD<PRD>();
	  if (prd.random() * 2.0f <= (float)prd.bounce) optixTerminateRay();
  }

  extern "C" __global__ void __anyhit__shadow()
  { /*! not going to be used */ }
  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    PRD &prd = *getPRD<PRD>();
	if (prd.bounce == 0) {
		prd.pixelColor = vec3f(1.f);
	}
	else {
		prd.hit = false;
	}
  }

  extern "C" __global__ void __miss__shadow()
  {
    // we didn't hit anything, so the light is visible
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = vec3f(1.f);
  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;
    
    PRD prd;
    prd.random.init(ix+optixLaunchParams.frame.size.x*iy,
                    optixLaunchParams.frame.frameID);
    prd.pixelColor = vec3f(0.f);
	prd.bounce = 0;


    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    int numPixelSamples = optixLaunchParams.numPixelSamples;//暂时为1

    vec3f pixelColor = 0.f;
    vec3f pixelNormal = 0.f;
    vec3f pixelAlbedo = 0.f;
    for (int sampleID=0;sampleID<numPixelSamples;sampleID++) {
      // normalized screen plane position, in [0,1]^2

      // iw: note for denoising that's not actually correct - if we
      // assume that the camera should only(!) cover the denoised
      // screen then the actual screen plane we shuld be using during
      // rendreing is slightly larger than [0,1]^2
      vec2f screen(vec2f(ix+prd.random(),iy+prd.random())
                   / vec2f(optixLaunchParams.frame.size));
      // screen
      //   = screen
      //   * vec2f(optixLaunchParams.frame.denoisedSize)
      //   * vec2f(optixLaunchParams.frame.size)
      //   - 0.5f*(vec2f(optixLaunchParams.frame.size)
      //           -
      //           vec2f(optixLaunchParams.frame.denoisedSize)
      //           );
      
      // generate ray direction
      vec3f rayDir = normalize(camera.direction
                               + (screen.x - 0.5f) * camera.horizontal
                               + (screen.y - 0.5f) * camera.vertical);

      optixTrace(optixLaunchParams.traversable,
                 camera.position,
                 rayDir,
                 0.f,    // tmin
                 1e20f,  // tmax
                 0.0f,   // rayTime
                 OptixVisibilityMask( 255 ),
				 OPTIX_RAY_FLAG_NONE,
                 RADIANCE_RAY_TYPE,            // SBT offset
                 RAY_TYPE_COUNT,               // SBT stride
                 RADIANCE_RAY_TYPE,            // missSBTIndex 
                 u0, u1 );
      pixelColor  += prd.pixelColor;
      pixelNormal += prd.pixelNormal;
      pixelAlbedo += prd.pixelAlbedo;
    }

    vec4f rgba(pixelColor/numPixelSamples,1.f);
    vec4f albedo(pixelAlbedo/numPixelSamples,1.f);
    vec4f normal(pixelNormal/numPixelSamples,1.f);

    // and write/accumulate to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    if (optixLaunchParams.frame.frameID > 0) {
      rgba
        += float(optixLaunchParams.frame.frameID)
        *  vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
      rgba /= (optixLaunchParams.frame.frameID+1.f);
    }
    optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
    optixLaunchParams.frame.albedoBuffer[fbIndex] = (float4)albedo;
    optixLaunchParams.frame.normalBuffer[fbIndex] = (float4)normal;
  }
  
} // ::osc
