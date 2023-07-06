#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "gdt/random/random.h"

using namespace osc;

#define NUM_LIGHT_SAMPLES 5//光源个数
#define MAXBOUNCE 5;

namespace osc {

  typedef gdt::LCG<16> Random;
  extern "C" __constant__ LaunchParams optixLaunchParams;
  extern "C" __constant__ float PI = 3.1415926;
  struct PRD {
    Random random;
    vec3f  pixelColor;
    vec3f  pixelNormal;
    vec3f  pixelAlbedo;
	bool done;
	bool first;
	vec3f origin;
	vec3f direction;
	vec3f attenuation;
  };
  static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, vec3f &p)
  {
	  // Uniformly sample disk.
	  const float r = u1;
	  const float phi = 2.0f * PI * u2;
	  p.x = r * gdt::cos(phi);
	  p.y = r * gdt::sin(phi);
	  p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x*p.x - p.y*p.y));
  }//极坐标转为直角坐标
  struct Onb
  {
	  __forceinline__ __device__ Onb(const vec3f &normal)
	  {
		  m_normal = normal;

		  if (fabs(m_normal.x) > fabs(m_normal.z))
		  {
			  m_binormal.x = -m_normal.y;
			  m_binormal.y = m_normal.x;
			  m_binormal.z = 0;
		  }
		  else
		  {
			  m_binormal.x = 0;
			  m_binormal.y = -m_normal.z;
			  m_binormal.z = m_normal.y;
		  }

		  m_binormal = normalize(m_binormal);
		  m_tangent = cross(m_binormal, m_normal);
	  }

	  __forceinline__ __device__ void inverse_transform(vec3f &p) const
	  {
		  p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;//将p从对应坐标系转化到世界坐标系
	  }

	  vec3f m_tangent;
	  vec3f m_binormal;
	  vec3f m_normal;//三个相互垂直的向量，构成一个类似坐标系的东西
  };
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
	  vec3f F = fresnelSchlick(h, v, F0);
	  vec3f former = specialmatch(color, kd) / PI;
	  float D = NDF(n, h, alpha);
	  float G = GeometrySmith(n, v, l, alpha);
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
    PRD &prd = *getPRD<PRD>();
	const int   primID = optixGetPrimitiveIndex();
	const vec3i index = sbtData.index[primID];
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;
	vec3f Kd;
	if (sbtData.Kd == vec3f(0.0)) Kd = vec3f(1.0);
	else Kd = sbtData.Kd;
	const vec3f Ks = sbtData.Ks;
	const vec3f &A = sbtData.vertex[index.x];
	const vec3f &B = sbtData.vertex[index.y];
	const vec3f &C = sbtData.vertex[index.z];
	vec3f Ng = gdt::cross(B - A, C - A);
	vec3f Ns = (sbtData.normal)
		? ((1.f - u - v) * sbtData.normal[index.x]
			+ u * sbtData.normal[index.y]
			+ v * sbtData.normal[index.z])
		: Ng;
	const vec3f rayDir = optixGetWorldRayDirection();

	if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
	Ng = normalize(Ng);

	if (dot(Ng, Ns) < 0.f)
		Ns -= 2.f*dot(Ng, Ns)*Ng;
	Ns = normalize(Ns);
	vec3f diffuseColor = sbtData.color;
	if (sbtData.hasTexture && sbtData.texcoord) {
		const vec2f tc
			= (1.f - u - v) * sbtData.texcoord[index.x]
			+ u * sbtData.texcoord[index.y]
			+ v * sbtData.texcoord[index.z];

		vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
		diffuseColor *= (vec3f)fromTexture;
	}
	vec3f dirlight = 0;
	const vec3f surfPos
		= (1.f - u - v) * sbtData.vertex[index.x]
		+ u * sbtData.vertex[index.y]
		+ v * sbtData.vertex[index.z];
	const vec3f view = gdt::normalize(optixLaunchParams.camera.position - surfPos);
	const int numLightSamples = NUM_LIGHT_SAMPLES;
	for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++) {
		for (int lightID = 0;lightID < optixLaunchParams.lightNum; lightID++) {
			const vec3f lightPos
				= optixLaunchParams.light[lightID].origin
				+ prd.random() * optixLaunchParams.light[lightID].du
				+ prd.random() * optixLaunchParams.light[lightID].dv;
			vec3f lightDir = lightPos - surfPos;
			float lightDist = gdt::length(lightDir);
			lightDir = normalize(lightDir);
			const float area = gdt::length(gdt::cross(optixLaunchParams.light[lightID].du, optixLaunchParams.light[lightID].dv));
			const vec3f Nl = gdt::normalize(gdt::cross(optixLaunchParams.light[lightID].du, optixLaunchParams.light[lightID].dv));
			const float NldotL = fabs(gdt::dot(Nl, lightDir));
			vec3f fr = BRDF(view, lightDir, Ns, vec3f(default_F0), default_alpha, diffuseColor, Kd, Ks);
			const float NdotL = dot(lightDir, Ns);
			if (NdotL >= 0.f && NldotL >= 0.f) {
				vec3f lightVisibility = 0.f;
				uint32_t u0, u1;
				packPointer(&lightVisibility, u0, u1);
				optixTrace(optixLaunchParams.traversable,
					surfPos + 1e-3f * Ng,
					lightDir,
					1e-3f,      // tmin
					lightDist * (1.f - 1e-3f),  // tmax
					0.0f,       // rayTime
					OptixVisibilityMask(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT
					| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
					| OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
					SHADOW_RAY_TYPE,            // SBT offset
					RAY_TYPE_COUNT,               // SBT stride
					SHADOW_RAY_TYPE,            // missSBTIndex 
					u0, u1);
				dirlight
					+= lightVisibility
					* optixLaunchParams.light[lightID].power
					* specialmatch(diffuseColor, fr)
					* (NdotL * NldotL * area / (lightDist * lightDist * numLightSamples))
					* optixLaunchParams.light[lightID].color;
			}
		}
	}
	prd.pixelColor += dirlight;
	vec3f random_sample = -Ns;
	while (gdt::dot(Ns, random_sample) <= 0) {
		float z1 = prd.random();
		float z2 = prd.random();
		cosine_sample_hemisphere(z1, z2, random_sample);
		Onb onb(Ns);
		onb.inverse_transform(random_sample);
	}
	vec3f lightDir = normalize(random_sample);
	prd.direction = lightDir;
	prd.origin = surfPos;
	vec3f fr = BRDF(view, lightDir, Ns, vec3f(default_F0), default_alpha, diffuseColor, Kd, Ks);
	float NdotL = gdt::dot(Ns, lightDir);
	prd.attenuation = specialmatch(NdotL * fr, prd.attenuation);
	if (prd.first) {
		prd.pixelAlbedo = diffuseColor;
		prd.pixelNormal = Ns;
		prd.first = false;
	}
  }
  
  extern "C" __global__ void __anyhit__radiance(){}

  extern "C" __global__ void __anyhit__shadow(){}
  
  extern "C" __global__ void __miss__radiance()
  {
    PRD &prd = *getPRD<PRD>();
	prd.pixelColor = vec3f(1.f);
	prd.done = true;
  }

  extern "C" __global__ void __miss__shadow()
  {
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
	prd.attenuation = 1.f;
	prd.done = false;
	prd.pixelColor = 0.f;
	prd.first = true;
	prd.pixelNormal = 0.f;
	prd.pixelAlbedo = 0.f;

    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    int numPixelSamples = optixLaunchParams.numPixelSamples;

    vec3f pixelColor = 0.f;
    vec3f pixelNormal = 0.f;
    vec3f pixelAlbedo = 0.f;
	int upperbound = MAXBOUNCE;
    for (int sampleID=0;sampleID<numPixelSamples;sampleID++) {
      vec2f screen(vec2f(ix+prd.random(),iy+prd.random())
                   / vec2f(optixLaunchParams.frame.size));
      vec3f rayDir = normalize(camera.direction
                               + (screen.x - 0.5f) * camera.horizontal
                               + (screen.y - 0.5f) * camera.vertical);
	  vec3f rayOrigin = camera.position;
	  int depth = 0;
	  vec3f result = 0.f;
	  while (!prd.done && depth <= upperbound) {
		  optixTrace(optixLaunchParams.traversable,
			  rayOrigin,
			  rayDir,
			  0.f,    // tmin
			  1e20f,  // tmax
			  0.0f,   // rayTime
			  OptixVisibilityMask(255),
			  OPTIX_RAY_FLAG_NONE,
			  RADIANCE_RAY_TYPE,            // SBT offset
			  RAY_TYPE_COUNT,               // SBT stride
			  RADIANCE_RAY_TYPE,            // missSBTIndex 
			  u0, u1);
		  rayOrigin = prd.origin;
		  rayDir = prd.direction;
		  result += prd.attenuation * prd.pixelColor;
		  depth++;
	  }
      pixelColor  += result;
      pixelNormal += prd.pixelNormal;
      pixelAlbedo += prd.pixelAlbedo;
    }

    vec4f rgba(pixelColor/numPixelSamples,1.f);
    vec4f albedo(pixelAlbedo/numPixelSamples,1.f);
    vec4f normal(pixelNormal/numPixelSamples,1.f);

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
