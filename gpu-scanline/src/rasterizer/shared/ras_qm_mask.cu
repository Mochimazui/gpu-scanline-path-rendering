
#include "ras_qm_mask.h"

#include <mochimazui/glpp.h>

namespace Mochimazui {

namespace QM_Mask_Sample_Position {

float mpvg_8_x[8] =
{
	-0.266471f, 0.353688f, -0.401679f, 0.488846f,
	0.122459f, -0.0344567f, -0.139007f, 0.207413f,
};

float mpvg_8_y[8] =
{
	0.164718f, 0.0396624f, -0.215021f, 0.429684f,
	0.282964f, -0.0841444f, -0.475235f, -0.328058f,
};

float mpvg_32_x[32] =  {
	0.18936400f, 0.31758200f, 0.00903428f, -0.21124700f,
	-0.36328000f, 0.33291100f, -0.49970400f, -0.43663000f,
	-0.26837500f, 0.37728700f, -0.18975300f, -0.48250600f,
	-0.13179200f, 0.49235500f, 0.42711400f, 0.37090100f,
	-0.31862800f, 0.02879450f, 0.04699840f, -0.16154700f,
	0.18745700f, -0.35758100f, 0.19744000f, 0.21454900f,
	-0.06650600f, 0.12811500f, 0.33646100f, 0.09921190f,
	-0.05305180f, -0.39892000f, -0.06095580f, -0.25435800f,
};

float mpvg_32_y[32] = {
	-0.34008000f, 0.40063000f, -0.37434000f, -0.08741820f,
	-0.43687600f, -0.04052820f, -0.09869600f, 0.12511500f,
	0.40422500f, 0.11086400f, -0.43411500f, -0.30842600f,
	-0.26574100f, 0.47500100f, 0.26635400f, -0.24554700f,
	-0.24432000f, 0.27314400f, -0.19358200f, 0.27910400f,
	0.13190500f, -0.03301080f, 0.29668700f, -0.15357100f,
	-0.05443540f, 0.47731600f, -0.43140000f, 0.00607759f,
	0.44544500f, 0.30706200f, 0.12207700f, 0.13223200f,
};

void sort_samples(std::vector<float2> &samples) {
	std::sort(samples.begin(), samples.end(), [](const float2 &a, const float2 &b) {
		return (a.y < b.y) || (a.y == b.y && a.x < b.x) ? true : false;
	});
}

std::vector<float2> mpvg_sample_position(int n_samples) {
	std::vector<float2> samples;
	samples.resize(n_samples);
	if (n_samples == 8) {
		for (int i = 0; i < 8; ++i) {
			samples[i].x = mpvg_8_x[i] + 0.5f;
			samples[i].y = 1.0f - (mpvg_8_y[i] + 0.5f);
		}
	}
	else if (n_samples == 32) {
		for (int i = 0; i < 32; ++i) {
			samples[i].x = mpvg_32_x[i] + 0.5f;
			samples[i].y = 1.0f - (mpvg_32_y[i] + 0.5f);
		}
	}
	else {
		throw std::runtime_error("mpvg_sample_position: unsupported sample number.");
	}
	sort_samples(samples);
	return samples;
}

std::vector<float2> gl_sample_position(int n_samples) {
	std::vector<float2> samples;

	GLPP::NamedTexture ttex;
	ttex.target(GLPP::Texture2DMultisample).create()
		.storage2DMultisample(n_samples, GL_RGBA8, 1, 1, GL_TRUE);

	GLPP::NamedFramebuffer tfbo;
	tfbo.create().texture2D(GL_COLOR_ATTACHMENT0, ttex, 0);

	tfbo.bind(GL_FRAMEBUFFER);

	int gl_n_samples;
	glGetIntegerv(GL_SAMPLES, &gl_n_samples);

	if (n_samples != gl_n_samples) {
		throw std::runtime_error("initQMMaskTable: incorrect sample number");
	}

	samples.resize(n_samples);
	for (int i = 0; i < n_samples; ++i) {
		glGetMultisamplefv(GL_SAMPLE_POSITION, i, (float*)(samples.data() + i));
	}

	tfbo.bind(GL_FRAMEBUFFER);

	tfbo.destroy();
	ttex.destroy();

	sort_samples(samples);
	return samples;
}

// -------- -------- -------- -------- -------- -------- -------- --------
// -------- -------- -------- -------- -------- -------- -------- --------

float vg_8_x[8] = {};
float vg_8_y[8] = {};

double vg_32_x[32] = {
	-0.49479166651144624000, 0.02604168653488159200, -0.24479165673255920000, 0.27604168653488159000, -0.36979167163372040000, 0.15104168653488159000, -0.09895834326744079600, 0.38020831346511841000, -0.43229166418313980000, 0.06770831346511840800, -0.18229165673255920000, 0.33854168653488159000, -0.30729167163372040000, 0.21354168653488159000, -0.05729165673255920400, 0.46354168653488159000, -0.46354166790843010000, 0.05729168653488159200, -0.19270834326744080000, 0.30729168653488159000, -0.31770832836627960000, 0.16145831346511841000, -0.06770834326744079600, 0.43229168653488159000, -0.40104166418313980000, 0.11979168653488159000, -0.13020834326744080000, 0.36979168653488159000, -0.27604167163372040000, 0.24479168653488159000, -0.02604165673255920400, 0.47395831346511841000,
};
double vg_32_y[32] = {
	-0.48437500000000000000, -0.45312500000000000000, -0.42187500000000000000, -0.39062500000000000000, -0.35937500000000000000, -0.32812500000000000000, -0.29687500000000000000, -0.26562500000000000000, -0.23437500000000000000, -0.20312500000000000000, -0.17187500000000000000, -0.14062500000000000000, -0.10937500000000000000, -0.07812500000000000000, -0.04687500000000000000, -0.01562500000000000000, 0.01562500000000000000, 0.04687500000000000000, 0.07812500000000000000, 0.10937500000000000000, 0.14062500000000000000, 0.17187500000000000000, 0.20312500000000000000, 0.23437500000000000000, 0.26562500000000000000, 0.29687500000000000000, 0.32812500000000000000, 0.35937500000000000000, 0.39062500000000000000, 0.42187500000000000000, 0.45312500000000000000, 0.48437500000000000000,
};

std::vector<float2> vg_sample_position(int i_n_samples) {

	std::vector<float2> fsample_8;
	std::vector<float2> fsample_32;

	fsample_8.resize(8);
	fsample_32.resize(32);

	for (int i = 0; i < 32; ++i) {
		fsample_32[i] = make_float2((float)(vg_32_x[i] + 0.5), (float)(vg_32_y[i] + 0.5));
	}

	{
		int2 isample_8[8];
		double gap = 1.0 / 8;
		double hgap = gap / 2;

		for (int i = 0; i < 8; ++i) {

			int bc = 0;
			int x = i;
			int y = 0;

			while (x) {
				y <<= 1;
				y |= x & 1;
				x >>= 1;
				++bc;
			}

			int scale = 1 << bc;
			double f = y / (float)scale;

			int yy = (int)(f * 8);

			auto &s = isample_8[i];
			s.x = yy;
			s.y = i;

			auto &sp = fsample_8[i];
			sp.x = (float)(s.x * gap + hgap);
			sp.y = (float)(s.y * gap + hgap);
		}

		sort_samples(fsample_8);
	}


	if (i_n_samples == 8) {
		return fsample_8;
	}
	else if(i_n_samples == 32) {
		return fsample_32;
	}
	else {
		throw std::runtime_error("vg_sample_position: only support 8x & 32x samples.");
	}
}

} // end of namespace QM_Mask_Sample_Position

} // end of namespace Mochimazui
