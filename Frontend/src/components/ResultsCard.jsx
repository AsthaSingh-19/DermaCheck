import { CheckCircle2, AlertCircle } from 'lucide-react';

/**
 * ResultsCard - Displays condition info: name, symptoms, causes, when to see doctor, care tips.
 * Used for both search results and scan analysis results.
 */
export default function ResultsCard({
  title,
  description,
  symptoms = [],
  causes = [],
  whenToSeeDoctor,
  careTips = [],
  isAnalysisResult = false,
}) {
  const sectionClass = 'space-y-2';
  const listClass = 'list-disc pl-5 text-warmGray text-sm space-y-1';

  return (
    <article
      className="rounded-soft-lg bg-white border border-beige-300/50 shadow-soft overflow-hidden animate-in fade-in duration-300"
      style={{ animationFillMode: 'backwards' }}
    >
      <div className="p-5 md:p-6 space-y-5">
        <div>
          <h2 className="text-xl font-poppins font-semibold text-darkBrown flex items-center gap-2">
            {isAnalysisResult ? (
              <AlertCircle className="w-6 h-6 text-pink-accent flex-shrink-0" />
            ) : (
              <CheckCircle2 className="w-6 h-6 text-pink-accent flex-shrink-0" />
            )}
            {title}
          </h2>
          {description && (
            <p className="mt-2 text-warmGray text-sm">{description}</p>
          )}
        </div>

        {symptoms.length > 0 && (
          <div className={sectionClass}>
            <h3 className="font-semibold text-darkBrown text-sm">Common symptoms</h3>
            <ul className={listClass}>
              {symptoms.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
        )}

        {causes.length > 0 && (
          <div className={sectionClass}>
            <h3 className="font-semibold text-darkBrown text-sm">Possible causes</h3>
            <ul className={listClass}>
              {causes.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}

        {whenToSeeDoctor && (
          <div className={`${sectionClass} p-3 rounded-soft bg-pink-light/40 border border-pink/20`}>
            <h3 className="font-semibold text-darkBrown text-sm">When to see a doctor</h3>
            <p className="text-warmGray text-sm mt-1">{whenToSeeDoctor}</p>
          </div>
        )}

        {careTips.length > 0 && (
          <div className={sectionClass}>
            <h3 className="font-semibold text-darkBrown text-sm">General care tips</h3>
            <ul className={listClass}>
              {careTips.map((t, i) => (
                <li key={i}>{t}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </article>
  );
}
